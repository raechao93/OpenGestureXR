using System;
using System.Collections;
using System.Threading;
using UnityEngine;
using UnityEngine.Networking;

namespace OpenGestureXR
{
    /// <summary>
    /// Connects to the gesture API server. Supports two modes:
    ///   - WebSocket-ish streaming (threaded HTTP polling at ~30fps — see note below)
    ///   - Classic HTTP polling (coroutine-based, configurable interval)
    ///
    /// NOTE: Unity doesn't ship a WebSocket client out of the box.
    /// For a real WS connection, drop in NativeWebSocket (https://github.com/endel/NativeWebSocket)
    /// and swap out WebSocketLoop(). The current impl fakes it with fast HTTP polling
    /// which is fine for local dev but you'd want real WS for production latency.
    /// </summary>
    public class GestureClient : MonoBehaviour
    {
        [Header("Server")]
        public string host = "localhost";
        public int port = 8000;

        [Header("Mode")]
        public bool useWebSocket = true;
        public float pollInterval = 0.1f; // only used in HTTP mode

        // events
        public static event Action<string, float> OnGestureReceived;
        public static event Action<GestureData[]> OnMultiHandReceived;

        [Serializable]
        public class GestureData
        {
            public string gesture;
            public float confidence;
            public string handedness;
        }

        [Serializable] private class MultiHandResponse
        {
            public string gesture;
            public float confidence;
            public GestureData[] hands;
        }

        [Serializable] private class SingleResponse
        {
            public string gesture;
            public float confidence;
        }

        private Thread _wsThread;
        private volatile bool _running;
        private volatile string _pendingMsg;

        void Start()
        {
            if (useWebSocket)
                StartStreaming();
            else
                StartCoroutine(PollLoop());
        }

        void OnDestroy() => _running = false;

        void Update()
        {
            // marshal WS messages back to main thread
            if (_pendingMsg == null) return;
            var json = _pendingMsg;
            _pendingMsg = null;
            HandleMessage(json);
        }

        void HandleMessage(string json)
        {
            // try multi-hand first, fall back to single
            try
            {
                var r = JsonUtility.FromJson<MultiHandResponse>(json);
                OnGestureReceived?.Invoke(r.gesture, r.confidence);
                if (r.hands is { Length: > 0 })
                    OnMultiHandReceived?.Invoke(r.hands);
            }
            catch
            {
                var r = JsonUtility.FromJson<SingleResponse>(json);
                OnGestureReceived?.Invoke(r.gesture, r.confidence);
            }
        }

        // --- "WebSocket" mode (really just threaded fast polling for now) ---

        void StartStreaming()
        {
            _running = true;
            _wsThread = new Thread(WebSocketLoop) { IsBackground = true };
            _wsThread.Start();
        }

        void WebSocketLoop()
        {
            // TODO: replace with NativeWebSocket for actual WS frames
            var url = $"http://{host}:{port}/gesture/multi";
            while (_running)
            {
                try
                {
                    using var wc = new System.Net.WebClient();
                    _pendingMsg = wc.DownloadString(url);
                    Thread.Sleep(33);
                }
                catch
                {
                    // server probably not up yet, back off a bit
                    Thread.Sleep(200);
                }
            }
        }

        // --- HTTP polling mode ---

        IEnumerator PollLoop()
        {
            var url = $"http://{host}:{port}/gesture";
            while (true)
            {
                using var req = UnityWebRequest.Get(url);
                yield return req.SendWebRequest();
                if (req.result == UnityWebRequest.Result.Success)
                    HandleMessage(req.downloadHandler.text);
                yield return new WaitForSeconds(pollInterval);
            }
        }
    }
}
