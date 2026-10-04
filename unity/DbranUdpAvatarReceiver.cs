using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

[DisallowMultipleComponent]
[RequireComponent(typeof(Animator))]
public sealed class DbranUdpAvatarReceiver : MonoBehaviour
{
    private const int PacketVersion = 1;
    private const int JointCount = 24;
    private const int RotationFloatCount = JointCount * 9;
    private const int HeaderSizeBytes = 44;
    private const int PacketSizeBytes = HeaderSizeBytes + RotationFloatCount * 4;
    private const uint FlagFullWindow = 1u << 0;

    [Header("UDP")]
    [SerializeField] private int listenPort = 9764;

    [Header("Avatar")]
    [SerializeField] private bool disableAnimatorAfterBinding = true;
    [SerializeField] private Vector3 modelToSceneEulerOffset = Vector3.zero;
    [SerializeField] private float translationScale = 1.0f;

    [Header("Optional smoothing")]
    [SerializeField] private float rotationSmoothing = 0.0f;
    [SerializeField] private float positionSmoothing = 0.0f;

    [Header("Runtime diagnostics")]
    [SerializeField] private bool isReceiving;
    [SerializeField] private uint lastSequence;
    [SerializeField] private int packetsReceived;
    [SerializeField] private int malformedPackets;
    [SerializeField] private int skippedSequences;
    [SerializeField] private int inputFrameIndex;
    [SerializeField] private int outputFrameIndex;

    private static readonly HumanBodyBones[] SmplToHumanBone =
    {
        HumanBodyBones.Hips,
        HumanBodyBones.LeftUpperLeg,
        HumanBodyBones.RightUpperLeg,
        HumanBodyBones.Spine,
        HumanBodyBones.LeftLowerLeg,
        HumanBodyBones.RightLowerLeg,
        HumanBodyBones.Chest,
        HumanBodyBones.LeftFoot,
        HumanBodyBones.RightFoot,
        HumanBodyBones.UpperChest,
        HumanBodyBones.LeftToes,
        HumanBodyBones.RightToes,
        HumanBodyBones.Neck,
        HumanBodyBones.LeftShoulder,
        HumanBodyBones.RightShoulder,
        HumanBodyBones.Head,
        HumanBodyBones.LeftUpperArm,
        HumanBodyBones.RightUpperArm,
        HumanBodyBones.LeftLowerArm,
        HumanBodyBones.RightLowerArm,
        HumanBodyBones.LeftHand,
        HumanBodyBones.RightHand,
        HumanBodyBones.LastBone,
        HumanBodyBones.LastBone
    };

    private sealed class PosePacket
    {
        public uint Sequence;
        public int InputFrame;
        public int OutputFrame;
        public ulong SourceHostUnixTimeNs;
        public uint Flags;
        public Vector3 RootTranslation;
        public readonly float[] RotationMatrices = new float[RotationFloatCount];
    }

    private Animator animatorComponent;
    private readonly Transform[] bones = new Transform[JointCount];
    private readonly Quaternion[] bindWorldRotations = new Quaternion[JointCount];

    private Vector3 initialAvatarPosition;
    private Quaternion sceneAlignment;
    private Quaternion inverseSceneAlignment;
    private bool translationOriginSet;
    private Vector3 translationOrigin;

    private UdpClient udpClient;
    private Thread receiveThread;
    private volatile bool receiveLoopRunning;

    private readonly object packetLock = new object();
    private PosePacket latestPacket;
    private bool hasLatestPacket;
    private uint lastAppliedSequence;
    private bool hasAppliedPacket;

    private void Start()
    {
        Application.runInBackground = true;

        animatorComponent = GetComponent<Animator>();
        if (animatorComponent.avatar == null ||
            !animatorComponent.avatar.isValid ||
            !animatorComponent.avatar.isHuman)
        {
            Debug.LogError(
                "D-BRAN receiver requires a valid Humanoid Animator avatar.",
                this
            );
            enabled = false;
            return;
        }

        animatorComponent.applyRootMotion = false;
        ResolveHumanoidBones();

        initialAvatarPosition = transform.position;
        sceneAlignment =
            transform.rotation * Quaternion.Euler(modelToSceneEulerOffset);
        inverseSceneAlignment = Quaternion.Inverse(sceneAlignment);

        if (disableAnimatorAfterBinding)
        {
            animatorComponent.enabled = false;
        }

        StartReceiver();
    }

    private void ResolveHumanoidBones()
    {
        for (int joint = 0; joint < JointCount; joint++)
        {
            HumanBodyBones humanBone = SmplToHumanBone[joint];
            if (humanBone == HumanBodyBones.LastBone)
            {
                continue;
            }

            Transform bone = animatorComponent.GetBoneTransform(humanBone);
            bones[joint] = bone;

            if (bone != null)
            {
                bindWorldRotations[joint] = bone.rotation;
            }
            else
            {
                Debug.LogWarning(
                    $"Humanoid bone {humanBone} is missing. " +
                    $"SMPL joint {joint} will be skipped.",
                    this
                );
            }
        }
    }

    private void StartReceiver()
    {
        try
        {
            udpClient = new UdpClient(listenPort);
            udpClient.Client.ReceiveTimeout = 500;
            receiveLoopRunning = true;
            receiveThread = new Thread(ReceiveLoop)
            {
                IsBackground = true,
                Name = "D-BRAN UDP receiver"
            };
            receiveThread.Start();
            isReceiving = true;
            Debug.Log(
                $"D-BRAN UDP receiver listening on 127.0.0.1:{listenPort}.",
                this
            );
        }
        catch (Exception exception)
        {
            Debug.LogError(
                $"Could not start D-BRAN UDP receiver: {exception}",
                this
            );
            enabled = false;
        }
    }

    private void ReceiveLoop()
    {
        IPEndPoint remote = new IPEndPoint(IPAddress.Any, 0);

        while (receiveLoopRunning)
        {
            try
            {
                byte[] bytes = udpClient.Receive(ref remote);
                if (!TryParsePacket(bytes, out PosePacket packet))
                {
                    Interlocked.Increment(ref malformedPackets);
                    continue;
                }

                lock (packetLock)
                {
                    latestPacket = packet;
                    hasLatestPacket = true;
                }

                Interlocked.Increment(ref packetsReceived);
            }
            catch (SocketException exception)
            {
                if (exception.SocketErrorCode != SocketError.TimedOut &&
                    receiveLoopRunning)
                {
                    Debug.LogWarning(
                        $"D-BRAN UDP receive error: {exception.Message}",
                        this
                    );
                }
            }
            catch (ObjectDisposedException)
            {
                break;
            }
            catch (Exception exception)
            {
                if (receiveLoopRunning)
                {
                    Debug.LogWarning(
                        $"D-BRAN UDP receiver exception: {exception}",
                        this
                    );
                }
            }
        }
    }

    private static bool TryParsePacket(byte[] bytes, out PosePacket packet)
    {
        packet = null;
        if (bytes == null || bytes.Length != PacketSizeBytes)
        {
            return false;
        }

        if (bytes[0] != (byte)'D' ||
            bytes[1] != (byte)'B' ||
            bytes[2] != (byte)'R' ||
            bytes[3] != (byte)'N')
        {
            return false;
        }

        int offset = 4;
        ushort version = ReadUInt16(bytes, ref offset);
        ushort jointCount = ReadUInt16(bytes, ref offset);
        if (version != PacketVersion || jointCount != JointCount)
        {
            return false;
        }

        PosePacket parsed = new PosePacket
        {
            Sequence = ReadUInt32(bytes, ref offset),
            InputFrame = ReadInt32(bytes, ref offset),
            OutputFrame = ReadInt32(bytes, ref offset),
            SourceHostUnixTimeNs = ReadUInt64(bytes, ref offset),
            Flags = ReadUInt32(bytes, ref offset)
        };

        parsed.RootTranslation = new Vector3(
            ReadSingle(bytes, ref offset),
            ReadSingle(bytes, ref offset),
            ReadSingle(bytes, ref offset)
        );

        for (int index = 0; index < RotationFloatCount; index++)
        {
            parsed.RotationMatrices[index] = ReadSingle(bytes, ref offset);
        }

        if ((parsed.Flags & FlagFullWindow) == 0)
        {
            return false;
        }

        packet = parsed;
        return true;
    }

    private void LateUpdate()
    {
        PosePacket packet = null;
        lock (packetLock)
        {
            if (hasLatestPacket)
            {
                packet = latestPacket;
                hasLatestPacket = false;
            }
        }

        if (packet == null)
        {
            return;
        }

        if (hasAppliedPacket)
        {
            uint difference = packet.Sequence - lastAppliedSequence;
            if (difference > 1)
            {
                skippedSequences += (int)(difference - 1);
            }
        }

        lastAppliedSequence = packet.Sequence;
        hasAppliedPacket = true;
        lastSequence = packet.Sequence;
        inputFrameIndex = packet.InputFrame;
        outputFrameIndex = packet.OutputFrame;

        ApplyRootTranslation(packet.RootTranslation);
        ApplyGlobalRotations(packet.RotationMatrices);
    }

    private void ApplyRootTranslation(Vector3 modelTranslation)
    {
        if (!translationOriginSet)
        {
            translationOrigin = modelTranslation;
            translationOriginSet = true;
        }

        Vector3 modelDelta =
            (modelTranslation - translationOrigin) * translationScale;
        Vector3 target = initialAvatarPosition + sceneAlignment * modelDelta;

        if (positionSmoothing <= 0.0f)
        {
            transform.position = target;
            return;
        }

        float blend = 1.0f - Mathf.Exp(-positionSmoothing * Time.deltaTime);
        transform.position = Vector3.Lerp(transform.position, target, blend);
    }

    private void ApplyGlobalRotations(float[] matrices)
    {
        for (int joint = 0; joint < JointCount; joint++)
        {
            Transform bone = bones[joint];
            if (bone == null)
            {
                continue;
            }

            int baseIndex = joint * 9;
            Vector3 up = new Vector3(
                matrices[baseIndex + 1],
                matrices[baseIndex + 4],
                matrices[baseIndex + 7]
            );
            Vector3 forward = new Vector3(
                matrices[baseIndex + 2],
                matrices[baseIndex + 5],
                matrices[baseIndex + 8]
            );

            if (up.sqrMagnitude < 1e-8f || forward.sqrMagnitude < 1e-8f)
            {
                continue;
            }

            Quaternion modelGlobal = Quaternion.LookRotation(
                forward.normalized,
                up.normalized
            );
            Quaternion worldDelta =
                sceneAlignment * modelGlobal * inverseSceneAlignment;
            Quaternion target = worldDelta * bindWorldRotations[joint];

            if (rotationSmoothing <= 0.0f)
            {
                bone.rotation = target;
            }
            else
            {
                float blend =
                    1.0f - Mathf.Exp(-rotationSmoothing * Time.deltaTime);
                bone.rotation = Quaternion.Slerp(bone.rotation, target, blend);
            }
        }
    }

    [ContextMenu("Reset Translation Origin")]
    public void ResetTranslationOrigin()
    {
        translationOriginSet = false;
        initialAvatarPosition = transform.position;
    }

    private void OnDisable()
    {
        StopReceiver();
    }

    private void OnDestroy()
    {
        StopReceiver();
    }

    private void StopReceiver()
    {
        if (!receiveLoopRunning && udpClient == null)
        {
            return;
        }

        receiveLoopRunning = false;
        isReceiving = false;
        try
        {
            udpClient?.Close();
        }
        catch
        {
        }
        udpClient = null;

        if (receiveThread != null &&
            receiveThread.IsAlive &&
            Thread.CurrentThread != receiveThread)
        {
            receiveThread.Join(1000);
        }
        receiveThread = null;
    }

    private static ushort ReadUInt16(byte[] bytes, ref int offset)
    {
        ushort value = BitConverter.ToUInt16(bytes, offset);
        offset += sizeof(ushort);
        return value;
    }

    private static uint ReadUInt32(byte[] bytes, ref int offset)
    {
        uint value = BitConverter.ToUInt32(bytes, offset);
        offset += sizeof(uint);
        return value;
    }

    private static int ReadInt32(byte[] bytes, ref int offset)
    {
        int value = BitConverter.ToInt32(bytes, offset);
        offset += sizeof(int);
        return value;
    }

    private static ulong ReadUInt64(byte[] bytes, ref int offset)
    {
        ulong value = BitConverter.ToUInt64(bytes, offset);
        offset += sizeof(ulong);
        return value;
    }

    private static float ReadSingle(byte[] bytes, ref int offset)
    {
        float value = BitConverter.ToSingle(bytes, offset);
        offset += sizeof(float);
        return value;
    }
}
