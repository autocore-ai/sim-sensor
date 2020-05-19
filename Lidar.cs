#region License
/*
* Copyright 2018 AutoCore
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#endregion

using System;
using System.Collections;
using System.IO;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace AutoCore.Sim.Sensor
{
    public class Lidar : MonoBehaviour
    {
        public bool LidarEnable { get; set; } = true;
        public Action<NativeArray<float4>> OnLidarData { get; set; }
        NativeArray<float4> data;
        NativeArray<float3> arrayLaserPosition;
        NativeArray<float3> arrayLaserRotation;
        NativeArray<RaycastHit> arrayRaycastHit;
        NativeArray<RaycastCommand> arrayRaycastCommand;
        const float distance = 50;

        int lastTimeStep;

        LidarGroupConfig lidarGroupConfig;
        int laserCount = 0;
        bool scanning = false;
        public string lidarConfig = "lidar.json";
        private void Start()
        {
            lidarGroupConfig = JsonUtility.FromJson<LidarGroupConfig>(File.ReadAllText(Path.Combine(Application.streamingAssetsPath, lidarConfig)));

            foreach (var lidarConfig in lidarGroupConfig.configs)
                laserCount += lidarConfig.laserVerticalAngle.Length * lidarConfig.horizontalResolution;
            arrayLaserPosition = new NativeArray<float3>(laserCount, Allocator.Persistent);
            arrayLaserRotation = new NativeArray<float3>(laserCount, Allocator.Persistent);
            int currentIndex = 0;
            for (int lc = 0; lc < lidarGroupConfig.configs.Length; lc++)
            {
                for (int la = 0; la < lidarGroupConfig.configs[lc].laserVerticalAngle.Length; la++)
                {
                    int horizonCount = lidarGroupConfig.configs[lc].horizontalResolution;
                    for (int i = 0; i < horizonCount; i++)
                    {
                        arrayLaserPosition[currentIndex] = lidarGroupConfig.configs[lc].position;
                        arrayLaserRotation[currentIndex++] = Quaternion.Euler(lidarGroupConfig.configs[lc].rotation) * Quaternion.Euler(lidarGroupConfig.configs[lc].laserVerticalAngle[la], 360.0f * i / horizonCount, 0) * Vector3.forward;
                    }
                }
            }
            arrayRaycastCommand = new NativeArray<RaycastCommand>(laserCount, Allocator.Persistent);
            arrayRaycastHit = new NativeArray<RaycastHit>(laserCount, Allocator.Persistent);
        }

        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.L))
                LidarEnable = !LidarEnable;
            if (!LidarEnable)
                return;

            if (scanning)
                return;

            var currentTimeStep = (int)(Time.time * lidarGroupConfig.frequency);
            if (currentTimeStep > lastTimeStep)
            {
                lastTimeStep = currentTimeStep;
                StartCoroutine(LidarScan());
            }
        }

        private IEnumerator LidarScan()
        {
            scanning = true;
            var jh1 = new ConfigRaycastCommand()
            {
                globalPosition = transform.position,
                position = arrayLaserPosition,
                globalRotation = transform.rotation,
                maxDistance = distance,
                direction = arrayLaserRotation,
                command = arrayRaycastCommand
            }.Schedule(arrayLaserRotation.Length, 64);

            var jh2 = RaycastCommand.ScheduleBatch(arrayRaycastCommand, arrayRaycastHit, 64, jh1);

            if (!jh2.IsCompleted)
            {
                yield return null;
            }

            var list = new NativeList<int>(arrayLaserRotation.Length, Allocator.TempJob);

            new FilterHitRays()
            {
                raycastHit = arrayRaycastHit
            }.ScheduleAppend(list, arrayLaserRotation.Length, 64, jh2).Complete();

            data = new NativeArray<float4>(list.Length, Allocator.TempJob);
            new GetLidarData()
            {
                nativeList = list,
                origin = transform.position + lidarGroupConfig.lidarOrigin,
                dataRotation = math.inverse(transform.rotation),
                direction = arrayLaserRotation,
                raycastHit = arrayRaycastHit,
                data = data
            }.Schedule(data.Length, 64).Complete();
            list.Dispose();
            OnLidarData?.Invoke(data);
            data.Dispose();

            scanning = false;
        }

        private void OnDestroy() => TryDispose();

        private void TryDispose()
        {
            if (arrayLaserPosition.IsCreated)
                arrayLaserPosition.Dispose();
            if (arrayLaserRotation.IsCreated)
                arrayLaserRotation.Dispose();
            if (arrayRaycastCommand.IsCreated)
                arrayRaycastCommand.Dispose();
            if (arrayRaycastHit.IsCreated)
                arrayRaycastHit.Dispose();
        }

        [BurstCompile]
        struct ConfigRaycastCommand : IJobParallelFor
        {
            [ReadOnly] public float maxDistance;
            [ReadOnly] public float3 globalPosition;
            [ReadOnly] public quaternion globalRotation;
            [ReadOnly] public NativeArray<float3> position;
            [ReadOnly] public NativeArray<float3> direction;
            public NativeArray<RaycastCommand> command;
            public void Execute(int index) => command[index] = new RaycastCommand(globalPosition + math.rotate(globalRotation, position[index]), math.rotate(globalRotation, direction[index]), maxDistance);
        }

        [BurstCompile]
        struct FilterHitRays : IJobParallelForFilter
        {
            [ReadOnly] public NativeArray<RaycastHit> raycastHit;
            public bool Execute(int index) => raycastHit[index].distance > 0;
        }

        [BurstCompile]
        struct GetLidarData : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int> nativeList;
            [ReadOnly] public Vector3 origin;
            [ReadOnly] public quaternion dataRotation;
            [ReadOnly] public NativeArray<float3> direction;
            [ReadOnly] public NativeArray<RaycastHit> raycastHit;
            public NativeArray<float4> data;
            public void Execute(int index)
            {
                int dataIndex = nativeList[index];
                float3 localPosition = math.rotate(dataRotation, raycastHit[dataIndex].point - origin);
                float density = math.dot(raycastHit[dataIndex].normal, direction[dataIndex]);
                data[index] = new float4(localPosition, math.abs(density) * (1 - math.length(localPosition) / distance));
            }
        }
    }
}