// File: xsens_stream_bridge.cpp
// Version: Six-sensor XDA synchronized acquisition with UDP streaming and CSV diagnostics
// Update rate: 60 Hz
// D-BRAN / TransPose order: LeftArm, RightArm, LeftLeg, RightLeg, Head, Hip
// UDP protocol: fixed-size little-endian binary datagrams on 127.0.0.1:9763
// Generated as a uniquely named revision to prevent accidental overwrite.

#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>

#pragma comment(lib, "Ws2_32.lib")

#include <xsensdeviceapi.h>
#include <xstypes/xsdatapacketptrarray.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>


// ------------------------------------------------------------
// Acquisition configuration.
// ------------------------------------------------------------
constexpr int UPDATE_RATE_HZ = 60;
constexpr int RADIO_CHANNEL = 19;
constexpr std::size_t SENSOR_COUNT = 6;
constexpr std::size_t MAX_SYNCHRONIZED_QUEUE_SIZE = 300;
constexpr int STABILIZATION_TIME_MS = 2000;

// UDP destination used by the Python D-BRAN receiver.
constexpr const char* UDP_HOST = "127.0.0.1";
constexpr unsigned short UDP_PORT = 9763;

// Fixed binary protocol sizes.
// Header format: <4sHHQQII
// Sensor format: <Iqq4d9d3d
constexpr std::uint16_t UDP_PROTOCOL_VERSION = 1;
constexpr std::uint32_t UDP_FRAME_COMPLETE_FLAG = 1;
constexpr std::size_t UDP_HEADER_SIZE = 32;
constexpr std::size_t UDP_SENSOR_RECORD_SIZE = 148;
constexpr std::size_t UDP_PACKET_SIZE =
    UDP_HEADER_SIZE + SENSOR_COUNT * UDP_SENSOR_RECORD_SIZE;


// ------------------------------------------------------------
// Fixed TransPose IMU order.
// ------------------------------------------------------------
struct RequiredSensor
{
    std::size_t transposeIndex;
    const char* bodyPart;
    const char* idSuffix;
};


const std::array<RequiredSensor, SENSOR_COUNT> REQUIRED_SENSORS =
{{
    {0, "LeftArm",  "244B"},
    {1, "RightArm", "244D"},
    {2, "LeftLeg",  "243A"},
    {3, "RightLeg", "2453"},
    {4, "Head",     "244C"},
    {5, "Hip",      "2452"}
}};


// ------------------------------------------------------------
// One sensor sample inside a synchronized six-sensor frame.
// ------------------------------------------------------------
struct SensorSample
{
    std::string sensorId;
    long long packetCounter = -1;
    long long sampleTimeFine = -1;

    std::array<double, 4> quaternion{};
    std::array<double, 9> rotationMatrix{};
    std::array<double, 3> acceleration{};
};


// ------------------------------------------------------------
// One synchronized frame in fixed TransPose order.
// ------------------------------------------------------------
struct SynchronizedFrame
{
    unsigned long long callbackSequence = 0;
    std::array<SensorSample, SENSOR_COUNT> sensors{};
};


// ------------------------------------------------------------
// Binary UDP serialization helpers.
//
// The bridge and Python receiver run on the same Windows x64
// computer. Numeric fields are serialized in little-endian form,
// matching Python struct formats that begin with '<'.
// ------------------------------------------------------------
template <typename T>
void appendPod(
    std::array<std::uint8_t, UDP_PACKET_SIZE>& packet,
    std::size_t& offset,
    const T& value)
{
    static_assert(
        std::is_trivially_copyable<T>::value,
        "UDP field must be trivially copyable.");

    if (offset + sizeof(T) > packet.size())
    {
        throw std::runtime_error(
            "Internal UDP packet serialization overflow.");
    }

    std::memcpy(
        packet.data() + offset,
        &value,
        sizeof(T));

    offset += sizeof(T);
}


void appendBytes(
    std::array<std::uint8_t, UDP_PACKET_SIZE>& packet,
    std::size_t& offset,
    const void* source,
    std::size_t byteCount)
{
    if (offset + byteCount > packet.size())
    {
        throw std::runtime_error(
            "Internal UDP packet serialization overflow.");
    }

    std::memcpy(
        packet.data() + offset,
        source,
        byteCount);

    offset += byteCount;
}


std::uint32_t deviceIdToUint32(
    const std::string& deviceId)
{
    std::size_t parsedCharacters = 0;
    const unsigned long value = std::stoul(
        deviceId,
        &parsedCharacters,
        16);

    if (parsedCharacters != deviceId.size())
    {
        throw std::runtime_error(
            "Could not parse Xsens device ID as hexadecimal: "
            + deviceId);
    }

    return static_cast<std::uint32_t>(value);
}


std::uint64_t currentUnixTimeNanoseconds()
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}


std::array<std::uint8_t, UDP_PACKET_SIZE> serializeUdpFrame(
    const SynchronizedFrame& frame)
{
    std::array<std::uint8_t, UDP_PACKET_SIZE> packet{};
    std::size_t offset = 0;

    const std::array<char, 4> magic = {{'D', 'B', 'R', 'N'}};
    appendBytes(packet, offset, magic.data(), magic.size());

    const std::uint16_t version = UDP_PROTOCOL_VERSION;
    const std::uint16_t sensorCount =
        static_cast<std::uint16_t>(SENSOR_COUNT);
    const std::uint64_t frameSequence =
        static_cast<std::uint64_t>(frame.callbackSequence);
    const std::uint64_t hostUnixTimeNs =
        currentUnixTimeNanoseconds();
    const std::uint32_t updateRateHz = UPDATE_RATE_HZ;
    const std::uint32_t flags = UDP_FRAME_COMPLETE_FLAG;

    appendPod(packet, offset, version);
    appendPod(packet, offset, sensorCount);
    appendPod(packet, offset, frameSequence);
    appendPod(packet, offset, hostUnixTimeNs);
    appendPod(packet, offset, updateRateHz);
    appendPod(packet, offset, flags);

    for (const SensorSample& sample : frame.sensors)
    {
        const std::uint32_t deviceId =
            deviceIdToUint32(sample.sensorId);
        const std::int64_t packetCounter =
            static_cast<std::int64_t>(sample.packetCounter);
        const std::int64_t sampleTimeFine =
            static_cast<std::int64_t>(sample.sampleTimeFine);

        appendPod(packet, offset, deviceId);
        appendPod(packet, offset, packetCounter);
        appendPod(packet, offset, sampleTimeFine);

        for (double value : sample.quaternion)
        {
            appendPod(packet, offset, value);
        }

        for (double value : sample.rotationMatrix)
        {
            appendPod(packet, offset, value);
        }

        for (double value : sample.acceleration)
        {
            appendPod(packet, offset, value);
        }
    }

    if (offset != packet.size())
    {
        throw std::runtime_error(
            "Internal UDP packet size mismatch.");
    }

    return packet;
}


// ------------------------------------------------------------
// RAII UDP sender.
// ------------------------------------------------------------
class UdpFrameSender
{
public:
    UdpFrameSender(
        const std::string& host,
        unsigned short port)
    {
        const int startupResult = WSAStartup(
            MAKEWORD(2, 2),
            &m_wsaData);

        if (startupResult != 0)
        {
            std::ostringstream error;
            error
                << "WSAStartup failed with error "
                << startupResult
                << ".";
            throw std::runtime_error(error.str());
        }

        m_wsaStarted = true;

        m_socket = socket(
            AF_INET,
            SOCK_DGRAM,
            IPPROTO_UDP);

        if (m_socket == INVALID_SOCKET)
        {
            const int errorCode = WSAGetLastError();
            cleanup();

            std::ostringstream error;
            error
                << "Failed to create UDP socket. WSA error "
                << errorCode
                << ".";
            throw std::runtime_error(error.str());
        }

        m_destination.sin_family = AF_INET;
        m_destination.sin_port = htons(port);

        const int addressResult = InetPtonA(
            AF_INET,
            host.c_str(),
            &m_destination.sin_addr);

        if (addressResult != 1)
        {
            cleanup();
            throw std::runtime_error(
                "Invalid IPv4 UDP destination: " + host);
        }
    }


    ~UdpFrameSender()
    {
        cleanup();
    }


    UdpFrameSender(const UdpFrameSender&) = delete;
    UdpFrameSender& operator=(const UdpFrameSender&) = delete;


    bool sendFrame(
        const SynchronizedFrame& frame,
        int& errorCode)
    {
        const std::array<std::uint8_t, UDP_PACKET_SIZE> packet =
            serializeUdpFrame(frame);

        const int sentBytes = sendto(
            m_socket,
            reinterpret_cast<const char*>(packet.data()),
            static_cast<int>(packet.size()),
            0,
            reinterpret_cast<const sockaddr*>(&m_destination),
            static_cast<int>(sizeof(m_destination)));

        if (sentBytes == SOCKET_ERROR)
        {
            errorCode = WSAGetLastError();
            return false;
        }

        if (sentBytes != static_cast<int>(packet.size()))
        {
            errorCode = -1;
            return false;
        }

        errorCode = 0;
        return true;
    }


private:
    void cleanup()
    {
        if (m_socket != INVALID_SOCKET)
        {
            closesocket(m_socket);
            m_socket = INVALID_SOCKET;
        }

        if (m_wsaStarted)
        {
            WSACleanup();
            m_wsaStarted = false;
        }
    }


    WSADATA m_wsaData{};
    bool m_wsaStarted = false;
    SOCKET m_socket = INVALID_SOCKET;
    sockaddr_in m_destination{};
};


// ------------------------------------------------------------
// Convert a string to uppercase for case-insensitive ID checks.
// ------------------------------------------------------------
std::string toUpperCopy(std::string value)
{
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char character)
        {
            return static_cast<char>(std::toupper(character));
        });

    return value;
}


// ------------------------------------------------------------
// Check whether a device ID ends with a required suffix.
// ------------------------------------------------------------
bool endsWithIgnoreCase(
    const std::string& value,
    const std::string& suffix)
{
    if (suffix.size() > value.size())
    {
        return false;
    }

    const std::string upperValue = toUpperCopy(value);
    const std::string upperSuffix = toUpperCopy(suffix);

    return upperValue.compare(
               upperValue.size() - upperSuffix.size(),
               upperSuffix.size(),
               upperSuffix) == 0;
}


// ------------------------------------------------------------
// Return the required sensor definition for a complete ID.
// ------------------------------------------------------------
const RequiredSensor* findRequiredSensor(
    const std::string& fullDeviceId)
{
    for (const RequiredSensor& sensor : REQUIRED_SENSORS)
    {
        if (endsWithIgnoreCase(
                fullDeviceId,
                sensor.idSuffix))
        {
            return &sensor;
        }
    }

    return nullptr;
}


// ------------------------------------------------------------
// Check whether a required suffix is present in connected IDs.
// ------------------------------------------------------------
bool containsRequiredSuffix(
    const std::set<std::string>& connectedDeviceIds,
    const std::string& requiredSuffix)
{
    for (const std::string& connectedDeviceId : connectedDeviceIds)
    {
        if (endsWithIgnoreCase(
                connectedDeviceId,
                requiredSuffix))
        {
            return true;
        }
    }

    return false;
}


// ------------------------------------------------------------
// Print Xsens port information.
// ------------------------------------------------------------
std::ostream& operator<<(
    std::ostream& out,
    const XsPortInfo& port)
{
    out << "Port: " << port.portNumber()
        << " (" << port.portName().toStdString() << ")"
        << " @ " << port.baudrate() << " Bd"
        << ", ID: " << port.deviceId().toString().toStdString();

    return out;
}


// ------------------------------------------------------------
// Print Xsens device information.
// ------------------------------------------------------------
std::ostream& operator<<(
    std::ostream& out,
    const XsDevice& device)
{
    out << "ID: "
        << device.deviceId().toString().toStdString()
        << " ("
        << device.productCode().toStdString()
        << ")";

    return out;
}


// ------------------------------------------------------------
// Wireless connectivity callback.
// ------------------------------------------------------------
class WirelessMasterCallback : public XsCallback
{
public:
    std::size_t connectedMtwCount() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_connectedMtwIds.size();
    }


    std::size_t connectedRequiredMtwCount() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        std::size_t count = 0;

        for (const RequiredSensor& sensor : REQUIRED_SENSORS)
        {
            if (containsRequiredSuffix(
                    m_connectedMtwIds,
                    sensor.idSuffix))
            {
                ++count;
            }
        }

        return count;
    }


    bool allRequiredMtwsConnected() const
    {
        return connectedRequiredMtwCount() == SENSOR_COUNT;
    }


    std::vector<RequiredSensor> missingRequiredSensors() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        std::vector<RequiredSensor> missingSensors;

        for (const RequiredSensor& sensor : REQUIRED_SENSORS)
        {
            if (!containsRequiredSuffix(
                    m_connectedMtwIds,
                    sensor.idSuffix))
            {
                missingSensors.push_back(sensor);
            }
        }

        return missingSensors;
    }


protected:
    void onConnectivityChanged(
        XsDevice* device,
        XsConnectivityState newState) override
    {
        if (device == nullptr)
        {
            return;
        }

        const std::string deviceId =
            device->deviceId().toString().toStdString();

        const RequiredSensor* requiredSensor =
            findRequiredSensor(deviceId);

        std::lock_guard<std::mutex> lock(m_mutex);

        switch (newState)
        {
        case XCS_Wireless:
            m_connectedMtwIds.insert(deviceId);

            std::cout
                << "\nEVENT: MTW Connected -> ID: "
                << deviceId;

            if (requiredSensor != nullptr)
            {
                std::cout
                    << " | Assignment: "
                    << requiredSensor->bodyPart;
            }
            else
            {
                std::cout
                    << " | Not in required sensor list";
            }

            std::cout << std::endl;
            break;

        case XCS_Disconnected:
            m_connectedMtwIds.erase(deviceId);

            std::cout
                << "\nEVENT: MTW Disconnected -> ID: "
                << deviceId;

            if (requiredSensor != nullptr)
            {
                std::cout
                    << " | Assignment: "
                    << requiredSensor->bodyPart;
            }

            std::cout << std::endl;
            break;

        case XCS_Rejected:
            m_connectedMtwIds.erase(deviceId);

            std::cout
                << "\nEVENT: MTW Rejected -> ID: "
                << deviceId;

            if (requiredSensor != nullptr)
            {
                std::cout
                    << " | Assignment: "
                    << requiredSensor->bodyPart;
            }

            std::cout << std::endl;
            break;

        default:
            return;
        }

        printStatusLocked();
    }


private:
    void printStatusLocked() const
    {
        std::size_t connectedRequiredCount = 0;

        std::cout
            << "Required sensor status:"
            << std::endl;

        for (const RequiredSensor& sensor : REQUIRED_SENSORS)
        {
            bool connected = false;
            std::string matchingDeviceId;

            for (const std::string& connectedDeviceId :
                 m_connectedMtwIds)
            {
                if (endsWithIgnoreCase(
                        connectedDeviceId,
                        sensor.idSuffix))
                {
                    connected = true;
                    matchingDeviceId = connectedDeviceId;
                    ++connectedRequiredCount;
                    break;
                }
            }

            std::cout
                << "  ["
                << (connected ? "OK" : "--")
                << "] "
                << sensor.transposeIndex
                << " "
                << sensor.bodyPart
                << " (*"
                << sensor.idSuffix
                << ")";

            if (connected)
            {
                std::cout
                    << " -> "
                    << matchingDeviceId;
            }

            std::cout << std::endl;
        }

        std::cout
            << "Required MTWs connected: "
            << connectedRequiredCount
            << "/"
            << SENSOR_COUNT
            << std::endl;

        std::cout
            << "All wireless MTWs connected: "
            << m_connectedMtwIds.size()
            << std::endl;
    }


    mutable std::mutex m_mutex;
    std::set<std::string> m_connectedMtwIds;
};


// ------------------------------------------------------------
// XDA multi-device callback.
//
// XDA calls onAllLiveDataAvailable from the wireless master with
// a group of packets for the connected child devices. This avoids
// manually aligning six independent application queues.
// ------------------------------------------------------------
class SynchronizedMtwCallback : public XsCallback
{
public:
    void setCaptureEnabled(bool enabled)
    {
        m_captureEnabled.store(
            enabled,
            std::memory_order_release);
    }


    void clear()
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        m_frameBuffer.clear();
        m_callbackAttempts = 0;
        m_completeFramesReceived = 0;
        m_incompleteCallbacks = 0;
        m_droppedBufferedFrames = 0;
        m_reportedMissedPackets = 0;
    }


    bool popOldestFrame(SynchronizedFrame& frame)
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_frameBuffer.empty())
        {
            return false;
        }

        frame = m_frameBuffer.front();
        m_frameBuffer.pop_front();

        return true;
    }


    std::size_t bufferedFrameCount() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_frameBuffer.size();
    }


    unsigned long long completeFramesReceived() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_completeFramesReceived;
    }


    unsigned long long callbackAttempts() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_callbackAttempts;
    }


    unsigned long long incompleteCallbacks() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_incompleteCallbacks;
    }


    unsigned long long droppedBufferedFrames() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_droppedBufferedFrames;
    }


    unsigned long long reportedMissedPackets() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_reportedMissedPackets;
    }


protected:
    void onAllLiveDataAvailable(
        XsDevicePtrArray* devices,
        const XsDataPacketPtrArray* packets) override
    {
        if (!m_captureEnabled.load(
                std::memory_order_acquire))
        {
            return;
        }

        if (devices == nullptr || packets == nullptr)
        {
            return;
        }

        const XsSize deviceCount = devices->size();
        const XsSize packetCount = packets->size();
        const XsSize pairCount =
            deviceCount < packetCount
                ? deviceCount
                : packetCount;

        SynchronizedFrame frame;
        std::array<bool, SENSOR_COUNT> sensorFound{};
        bool duplicateSensor = false;

        for (XsSize i = 0;
             i < pairCount;
             ++i)
        {
            XsDevice* device = (*devices)[i];
            const XsDataPacket* packet = (*packets)[i];

            if (device == nullptr || packet == nullptr)
            {
                continue;
            }

            const std::string deviceId =
                device->deviceId().toString().toStdString();

            const RequiredSensor* requiredSensor =
                findRequiredSensor(deviceId);

            if (requiredSensor == nullptr)
            {
                continue;
            }

            const std::size_t sensorIndex =
                requiredSensor->transposeIndex;

            if (sensorFound[sensorIndex])
            {
                duplicateSensor = true;
                continue;
            }

            if (!packet->containsOrientation() ||
                !packet->containsCalibratedAcceleration())
            {
                continue;
            }

            SensorSample& sample =
                frame.sensors[sensorIndex];

            sample.sensorId = deviceId;

            sample.packetCounter =
                packet->containsPacketCounter()
                    ? static_cast<long long>(
                          packet->packetCounter())
                    : -1;

            sample.sampleTimeFine =
                packet->containsSampleTimeFine()
                    ? static_cast<long long>(
                          packet->sampleTimeFine())
                    : -1;

            const XsQuaternion quaternion =
                packet->orientationQuaternion();

            sample.quaternion =
            {
                quaternion.w(),
                quaternion.x(),
                quaternion.y(),
                quaternion.z()
            };

            const XsMatrix rotationMatrix =
                packet->orientationMatrix();

            sample.rotationMatrix =
            {
                rotationMatrix.value(0, 0),
                rotationMatrix.value(0, 1),
                rotationMatrix.value(0, 2),

                rotationMatrix.value(1, 0),
                rotationMatrix.value(1, 1),
                rotationMatrix.value(1, 2),

                rotationMatrix.value(2, 0),
                rotationMatrix.value(2, 1),
                rotationMatrix.value(2, 2)
            };

            const XsVector acceleration =
                packet->calibratedAcceleration();

            sample.acceleration =
            {
                acceleration[0],
                acceleration[1],
                acceleration[2]
            };

            sensorFound[sensorIndex] = true;
        }

        const bool completeFrame =
            !duplicateSensor &&
            std::all_of(
                sensorFound.begin(),
                sensorFound.end(),
                [](bool found)
                {
                    return found;
                });

        std::lock_guard<std::mutex> lock(m_mutex);

        ++m_callbackAttempts;

        if (!completeFrame)
        {
            ++m_incompleteCallbacks;
            return;
        }

        ++m_completeFramesReceived;
        frame.callbackSequence = m_callbackAttempts;

        if (m_frameBuffer.size() >=
            MAX_SYNCHRONIZED_QUEUE_SIZE)
        {
            m_frameBuffer.pop_front();
            ++m_droppedBufferedFrames;
        }

        m_frameBuffer.push_back(frame);
    }


    void onMissedPackets(
        XsDevice*,
        int count,
        int,
        int) override
    {
        if (count <= 0)
        {
            return;
        }

        std::lock_guard<std::mutex> lock(m_mutex);

        m_reportedMissedPackets +=
            static_cast<unsigned long long>(count);
    }


private:
    std::atomic<bool> m_captureEnabled{false};

    mutable std::mutex m_mutex;
    std::deque<SynchronizedFrame> m_frameBuffer;

    // Increments on every synchronized-callback attempt, complete or not.
    // frame.callbackSequence is assigned from this counter (not from
    // m_completeFramesReceived), so a gap in the sequence numbers the
    // Python side sees corresponds to a real missed/incomplete slot,
    // instead of being invisible because only successes were counted.
    unsigned long long m_callbackAttempts = 0;

    unsigned long long m_completeFramesReceived = 0;
    unsigned long long m_incompleteCallbacks = 0;
    unsigned long long m_droppedBufferedFrames = 0;
    unsigned long long m_reportedMissedPackets = 0;
};


// ------------------------------------------------------------
// Main.
// ------------------------------------------------------------
int main()
{
    WirelessMasterCallback wirelessMasterCallback;
    SynchronizedMtwCallback synchronizedMtwCallback;
    UdpFrameSender udpSender(UDP_HOST, UDP_PORT);

    std::cout
        << "Xsens MTw2 Six-Sensor UDP Stream Bridge"
        << std::endl;

    std::cout
        << "D-BRAN / TransPose order: LeftArm, RightArm, LeftLeg, "
        << "RightLeg, Head, Hip"
        << std::endl;

    std::cout
        << "UDP destination: "
        << UDP_HOST
        << ":"
        << UDP_PORT
        << " | protocol v"
        << UDP_PROTOCOL_VERSION
        << " | datagram size "
        << UDP_PACKET_SIZE
        << " bytes"
        << std::endl;

    std::cout
        << "Scanning for Awinda wireless master..."
        << std::endl;

    XsControl* control = XsControl::construct();

    if (control == nullptr)
    {
        std::cerr
            << "Failed to construct XsControl."
            << std::endl;

        return 1;
    }

    try
    {
        // ----------------------------------------------------
        // Scan connected Xsens devices.
        // ----------------------------------------------------
        const XsPortInfoArray detectedDevices =
            XsScanner::scanPorts();

        if (detectedDevices.empty())
        {
            throw std::runtime_error(
                "No Xsens devices found.");
        }

        std::cout
            << "\nDetected devices:"
            << std::endl;

        for (XsPortInfoArray::const_iterator it =
                 detectedDevices.begin();
             it != detectedDevices.end();
             ++it)
        {
            std::cout
                << "  "
                << *it
                << std::endl;
        }

        // ----------------------------------------------------
        // Find the Awinda wireless master.
        // ----------------------------------------------------
        XsPortInfoArray::const_iterator wirelessMasterPort =
            detectedDevices.begin();

        while (
            wirelessMasterPort != detectedDevices.end() &&
            !wirelessMasterPort->deviceId().isWirelessMaster())
        {
            ++wirelessMasterPort;
        }

        if (wirelessMasterPort == detectedDevices.end())
        {
            throw std::runtime_error(
                "No Awinda wireless master found.");
        }

        std::cout
            << "\nWireless master found:"
            << std::endl;

        std::cout
            << "  "
            << *wirelessMasterPort
            << std::endl;

        // ----------------------------------------------------
        // Open the wireless master port.
        // ----------------------------------------------------
        std::cout
            << "\nOpening wireless master port..."
            << std::endl;

        if (!control->openPort(
                wirelessMasterPort->portName().toStdString(),
                wirelessMasterPort->baudrate()))
        {
            std::ostringstream error;

            error
                << "Failed to open port: "
                << *wirelessMasterPort;

            throw std::runtime_error(error.str());
        }

        std::cout
            << "Port opened successfully."
            << std::endl;

        XsDevicePtr wirelessMasterDevice =
            control->device(
                wirelessMasterPort->deviceId());

        if (wirelessMasterDevice == nullptr)
        {
            throw std::runtime_error(
                "Failed to create wireless master XsDevice.");
        }

        std::cout
            << "\nWireless master device:"
            << std::endl;

        std::cout
            << "  "
            << *wirelessMasterDevice
            << std::endl;

        // ----------------------------------------------------
        // Configure the wireless master.
        // ----------------------------------------------------
        std::cout
            << "\nSetting config mode..."
            << std::endl;

        if (!wirelessMasterDevice->gotoConfig())
        {
            throw std::runtime_error(
                "Failed to enter config mode.");
        }

        std::cout
            << "Config mode set successfully."
            << std::endl;

        wirelessMasterDevice->addCallbackHandler(
            &wirelessMasterCallback);

        const XsIntArray supportedUpdateRates =
            wirelessMasterDevice->supportedUpdateRates();

        if (supportedUpdateRates.empty())
        {
            throw std::runtime_error(
                "No supported update rates reported.");
        }

        bool updateRateSupported = false;

        std::cout
            << "\nSupported update rates: ";

        for (XsIntArray::const_iterator it =
                 supportedUpdateRates.begin();
             it != supportedUpdateRates.end();
             ++it)
        {
            std::cout
                << *it
                << " ";

            if (*it == UPDATE_RATE_HZ)
            {
                updateRateSupported = true;
            }
        }

        std::cout << std::endl;

        if (!updateRateSupported)
        {
            std::ostringstream error;

            error
                << "Required update rate "
                << UPDATE_RATE_HZ
                << " Hz is not supported.";

            throw std::runtime_error(error.str());
        }

        std::cout
            << "Selected update rate: "
            << UPDATE_RATE_HZ
            << " Hz"
            << std::endl;

        if (!wirelessMasterDevice->setUpdateRate(
                UPDATE_RATE_HZ))
        {
            throw std::runtime_error(
                "Failed to set update rate.");
        }

        if (wirelessMasterDevice->isRadioEnabled())
        {
            std::cout
                << "\nRadio already enabled. Disabling..."
                << std::endl;

            if (!wirelessMasterDevice->disableRadio())
            {
                throw std::runtime_error(
                    "Failed to disable radio.");
            }
        }

        std::cout
            << "\nEnabling radio channel "
            << RADIO_CHANNEL
            << "..."
            << std::endl;

        if (!wirelessMasterDevice->enableRadio(
                RADIO_CHANNEL))
        {
            throw std::runtime_error(
                "Failed to enable radio.");
        }

        std::cout
            << "Radio enabled successfully."
            << std::endl;

        // ----------------------------------------------------
        // Wait for the exact six required sensors.
        // ----------------------------------------------------
        std::cout
            << "\nRequired MTw assignments in TransPose order:"
            << std::endl;

        for (const RequiredSensor& sensor : REQUIRED_SENSORS)
        {
            std::cout
                << "  ["
                << sensor.transposeIndex
                << "] "
                << sensor.bodyPart
                << " -> ID suffix "
                << sensor.idSuffix
                << std::endl;
        }

        std::cout
            << "\nTurn on the six required MTw2 sensors."
            << std::endl;

        std::cout
            << "Press ENTER only when the status shows 6/6."
            << std::endl;

        std::cin.get();

        if (!wirelessMasterCallback.allRequiredMtwsConnected())
        {
            const std::vector<RequiredSensor> missingSensors =
                wirelessMasterCallback.missingRequiredSensors();

            std::ostringstream error;

            error
                << "Cannot start measurement. Missing required MTWs:";

            for (const RequiredSensor& sensor : missingSensors)
            {
                error
                    << " "
                    << sensor.bodyPart
                    << "(*"
                    << sensor.idSuffix
                    << ")";
            }

            throw std::runtime_error(error.str());
        }

        std::cout
            << "\nAll six required MTWs are connected."
            << std::endl;

        // ----------------------------------------------------
        // Attach XDA's multi-device callback to the master.
        // Capture remains disabled during startup.
        // ----------------------------------------------------
        synchronizedMtwCallback.setCaptureEnabled(false);
        synchronizedMtwCallback.clear();

        wirelessMasterDevice->addCallbackHandler(
            &synchronizedMtwCallback);

        // ----------------------------------------------------
        // Enter measurement mode.
        // ----------------------------------------------------

        std::cout << "\nMeasurement will start in 5 seconds." << std::endl;

        for (int remaining = 5; remaining > 0; --remaining)
        {
            std::cout
                << "\rStarting measurement in "
                << remaining
                << " seconds... "
                << std::flush;

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        std::cout << "\rStarting measurement now.          " << std::endl;

        std::cout
            << "\nStarting measurement mode..."
            << std::endl;

        if (!wirelessMasterDevice->gotoMeasurement())
        {
            throw std::runtime_error(
                "Failed to enter measurement mode.");
        }

        std::cout
            << "Measurement mode started successfully."
            << std::endl;

        // ----------------------------------------------------
        // Verify the six device instances in fixed order.
        // ----------------------------------------------------
        const XsDeviceIdArray allDeviceIds =
            control->deviceIds();

        std::array<XsDevicePtr, SENSOR_COUNT> orderedDevices{};
        std::array<bool, SENSOR_COUNT> orderedDeviceFound{};

        for (XsDeviceIdArray::const_iterator it =
                 allDeviceIds.begin();
             it != allDeviceIds.end();
             ++it)
        {
            if (!it->isMtw())
            {
                continue;
            }

            const std::string fullDeviceId =
                it->toString().toStdString();

            const RequiredSensor* requiredSensor =
                findRequiredSensor(fullDeviceId);

            if (requiredSensor == nullptr)
            {
                continue;
            }

            const std::size_t index =
                requiredSensor->transposeIndex;

            if (orderedDeviceFound[index])
            {
                std::ostringstream error;

                error
                    << "More than one MTw matches required suffix "
                    << requiredSensor->idSuffix
                    << ".";

                throw std::runtime_error(error.str());
            }

            orderedDevices[index] =
                control->device(*it);

            if (orderedDevices[index] == nullptr)
            {
                std::ostringstream error;

                error
                    << "Failed to create XsDevice for required MTw "
                    << fullDeviceId
                    << ".";

                throw std::runtime_error(error.str());
            }

            orderedDeviceFound[index] = true;
        }

        if (!std::all_of(
                orderedDeviceFound.begin(),
                orderedDeviceFound.end(),
                [](bool found)
                {
                    return found;
                }))
        {
            throw std::runtime_error(
                "Not all six required MTw device instances are available.");
        }

        std::cout
            << "\nRequired MTw device instances:"
            << std::endl;

        for (std::size_t i = 0;
             i < SENSOR_COUNT;
             ++i)
        {
            std::cout
                << "  ["
                << i
                << "] "
                << REQUIRED_SENSORS[i].bodyPart
                << " -> "
                << *orderedDevices[i]
                << std::endl;
        }

        // ----------------------------------------------------
        // Startup stabilization and application-buffer reset.
        //
        // The shared capture gate prevents any synchronized XDA
        // frames from entering our queue during this interval.
        // ----------------------------------------------------
        std::cout
            << "\nStabilizing synchronized stream for "
            << STABILIZATION_TIME_MS
            << " ms..."
            << std::endl;

        std::this_thread::sleep_for(
            std::chrono::milliseconds(
                STABILIZATION_TIME_MS));

        synchronizedMtwCallback.clear();

        std::cout
            << "Application synchronized-frame buffer cleared."
            << std::endl;

        // ----------------------------------------------------
        // Create CSV file before opening the capture gate.
        // ----------------------------------------------------
        std::filesystem::create_directories(
            "data/logs");

        const std::string csvPath =
            "data/logs/xsens_stream_bridge_log.csv";

        std::ofstream csvFile(
            csvPath,
            std::ios::out);

        if (!csvFile.is_open())
        {
            throw std::runtime_error(
                "Failed to create CSV log file.");
        }

        // ----------------------------------------------------
        // Keep the same CSV data fields used previously.
        // processed_packet_count is now the synchronized frame
        // index and is shared by the six rows of that frame.
        // ----------------------------------------------------
        csvFile
            << "processed_packet_count,"
            << "sensor_index,"
            << "sensor_id,"
            << "packet_counter,"
            << "sample_time_fine,"

            << "quat_w,"
            << "quat_x,"
            << "quat_y,"
            << "quat_z,"

            << "r00,"
            << "r01,"
            << "r02,"
            << "r10,"
            << "r11,"
            << "r12,"
            << "r20,"
            << "r21,"
            << "r22,"

            << "acc_x,"
            << "acc_y,"
            << "acc_z"
            << "\n";

        unsigned long long synchronizedFramesProcessed = 0;
        unsigned long long loggedRows = 0;
        unsigned long long udpFramesSent = 0;
        unsigned long long udpSendFailures = 0;

        std::cout
            << "\nCSV diagnostics ready:"
            << std::endl;

        std::cout
            << "  "
            << csvPath
            << std::endl;

        std::cout
            << "Opening synchronized capture gate."
            << std::endl;

        synchronizedMtwCallback.setCaptureEnabled(true);

        std::atomic<bool> stopRequested(false);

        std::cout
            << "\nSynchronized six-sensor UDP streaming started at "
            << UPDATE_RATE_HZ
            << " Hz."
            << std::endl;

        std::cout
            << "Python receiver: "
            << UDP_HOST
            << ":"
            << UDP_PORT
            << std::endl;

        std::cout
            << "The 26-frame online window is maintained by dbran.pipeline."
            << std::endl;

        std::cout
            << "Press ENTER to stop measurement."
            << std::endl;

        std::thread inputThread(
            [&stopRequested]()
            {
                std::cin.get();
                stopRequested.store(true);
            });

        // ----------------------------------------------------
        // Process one synchronized frame.
        // ----------------------------------------------------
        const auto processFrame =
            [&](const SynchronizedFrame& frame)
            {
                ++synchronizedFramesProcessed;

                int udpErrorCode = 0;

                if (udpSender.sendFrame(
                        frame,
                        udpErrorCode))
                {
                    ++udpFramesSent;
                }
                else
                {
                    ++udpSendFailures;

                    if (udpSendFailures <= 5 ||
                        udpSendFailures % 100 == 0)
                    {
                        std::cerr
                            << "UDP send failure "
                            << udpSendFailures
                            << " | error code "
                            << udpErrorCode
                            << std::endl;
                    }
                }

                for (std::size_t sensorIndex = 0;
                     sensorIndex < SENSOR_COUNT;
                     ++sensorIndex)
                {
                    const SensorSample& sample =
                        frame.sensors[sensorIndex];

                    csvFile
                        << synchronizedFramesProcessed << ","
                        << sensorIndex << ","
                        << sample.sensorId << ","
                        << sample.packetCounter << ","
                        << sample.sampleTimeFine << ","

                        << std::fixed
                        << std::setprecision(9)

                        << sample.quaternion[0] << ","
                        << sample.quaternion[1] << ","
                        << sample.quaternion[2] << ","
                        << sample.quaternion[3] << ","

                        << sample.rotationMatrix[0] << ","
                        << sample.rotationMatrix[1] << ","
                        << sample.rotationMatrix[2] << ","
                        << sample.rotationMatrix[3] << ","
                        << sample.rotationMatrix[4] << ","
                        << sample.rotationMatrix[5] << ","
                        << sample.rotationMatrix[6] << ","
                        << sample.rotationMatrix[7] << ","
                        << sample.rotationMatrix[8] << ","

                        << sample.acceleration[0] << ","
                        << sample.acceleration[1] << ","
                        << sample.acceleration[2]
                        << "\n";

                    ++loggedRows;
                }

                if (loggedRows % 600 == 0)
                {
                    csvFile.flush();
                }

                if (synchronizedFramesProcessed % 25 == 0)
                {
                    std::cout
                        << "\nFrame "
                        << synchronizedFramesProcessed
                        << " | UDP sent: "
                        << udpFramesSent
                        << " | UDP failures: "
                        << udpSendFailures
                        << std::endl;

                    for (std::size_t sensorIndex = 0;
                         sensorIndex < SENSOR_COUNT;
                         ++sensorIndex)
                    {
                        const SensorSample& sample =
                            frame.sensors[sensorIndex];

                        std::cout
                            << "  ["
                            << sensorIndex
                            << "] "
                            << REQUIRED_SENSORS[sensorIndex].bodyPart
                            << ", ID: "
                            << sample.sensorId
                            << ", PacketCounter: "
                            << sample.packetCounter
                            << ", Acc: ["
                            << std::fixed
                            << std::setprecision(3)
                            << sample.acceleration[0]
                            << ", "
                            << sample.acceleration[1]
                            << ", "
                            << sample.acceleration[2]
                            << "]"
                            << std::endl;
                    }
                }
            };

        // ----------------------------------------------------
        // Main synchronized-frame processing loop.
        // ----------------------------------------------------
        while (!stopRequested.load())
        {
            bool frameProcessed = false;
            SynchronizedFrame frame;

            while (synchronizedMtwCallback.popOldestFrame(
                       frame))
            {
                frameProcessed = true;
                processFrame(frame);
            }

            if (!frameProcessed)
            {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(1));
            }
        }

        // ----------------------------------------------------
        // Close the shared capture gate and drain frames that
        // were already queued before the stop request.
        // ----------------------------------------------------
        synchronizedMtwCallback.setCaptureEnabled(false);

        // Allow any callback already in progress to finish before draining.
        std::this_thread::sleep_for(
            std::chrono::milliseconds(50));

        SynchronizedFrame remainingFrame;

        while (synchronizedMtwCallback.popOldestFrame(
                   remainingFrame))
        {
            processFrame(remainingFrame);
        }

        inputThread.join();

        csvFile.flush();
        csvFile.close();

        std::cout
            << "\nUDP streaming and CSV diagnostics stopped."
            << std::endl;

        std::cout
            << "Synchronized frames processed: "
            << synchronizedFramesProcessed
            << std::endl;

        std::cout
            << "Rows written: "
            << loggedRows
            << " ("
            << SENSOR_COUNT
            << " rows per synchronized frame)"
            << std::endl;

        std::cout
            << "UDP frames sent: "
            << udpFramesSent
            << std::endl;

        std::cout
            << "UDP send failures: "
            << udpSendFailures
            << std::endl;

        std::cout
            << "Synchronized callback attempts: "
            << synchronizedMtwCallback.callbackAttempts()
            << " (this is the sequence-number space; gaps in it are what "
            << "the Python side now detects)"
            << std::endl;

        std::cout
            << "Incomplete multi-device callbacks: "
            << synchronizedMtwCallback.incompleteCallbacks()
            << std::endl;

        std::cout
            << "Application queue frames dropped: "
            << synchronizedMtwCallback.droppedBufferedFrames()
            << std::endl;

        std::cout
            << "Missed packets reported by XDA: "
            << synchronizedMtwCallback.reportedMissedPackets()
            << std::endl;

        std::cout
            << "File: "
            << csvPath
            << std::endl;

        // ----------------------------------------------------
        // Remove the synchronized callback before shutdown.
        // ----------------------------------------------------
        wirelessMasterDevice->removeCallbackHandler(
            &synchronizedMtwCallback);

        // ----------------------------------------------------
        // Return to configuration mode and disable radio.
        // ----------------------------------------------------
        std::cout
            << "\nReturning to config mode..."
            << std::endl;

        if (!wirelessMasterDevice->gotoConfig())
        {
            throw std::runtime_error(
                "Failed to return to config mode.");
        }

        std::cout
            << "Config mode restored successfully."
            << std::endl;

        std::cout
            << "\nDisabling radio..."
            << std::endl;

        if (!wirelessMasterDevice->disableRadio())
        {
            throw std::runtime_error(
                "Failed to disable radio.");
        }

        std::cout
            << "Radio disabled successfully."
            << std::endl;

        wirelessMasterDevice->removeCallbackHandler(
            &wirelessMasterCallback);
    }
    catch (const std::exception& ex)
    {
        synchronizedMtwCallback.setCaptureEnabled(false);

        std::cerr
            << "\nError: "
            << ex.what()
            << std::endl;

        control->close();
        return 1;
    }

    control->close();

    std::cout
        << "\nDone."
        << std::endl;

    return 0;
}
