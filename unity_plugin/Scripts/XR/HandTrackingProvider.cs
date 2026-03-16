using UnityEngine;

namespace OpenGestureXR.XR
{
    public enum Handedness { Left, Right }

    /// <summary>
    /// Base class for hand tracking backends. Implement one per platform
    /// (Quest, Pico, etc.) and drop it on the same GameObject as GestureClient.
    ///
    /// For now we only have the server-based provider that gets data from
    /// the Python API. Native OpenXR providers are on the roadmap.
    /// </summary>
    public abstract class HandTrackingProvider : MonoBehaviour
    {
        public abstract bool IsTracking { get; }
        public abstract Vector3[] GetJointPositions(Handedness hand);
        public abstract Quaternion[] GetJointRotations(Handedness hand);
    }

    /// <summary>
    /// Default provider — just reflects whether the Python server is detecting a hand.
    /// Joint positions aren't populated yet since the server only sends gesture labels,
    /// not raw landmarks. That's a TODO for when we add the landmark streaming endpoint.
    /// </summary>
    public class ServerHandTrackingProvider : HandTrackingProvider
    {
        public override bool IsTracking => _tracking;

        bool _tracking;
        readonly Vector3[] _positions = new Vector3[21];

        void OnEnable()  => GestureClient.OnGestureReceived += OnGesture;
        void OnDisable() => GestureClient.OnGestureReceived -= OnGesture;

        void OnGesture(string gesture, float conf)
        {
            _tracking = gesture != "none";
        }

        // TODO: populate these from a /ws/landmarks endpoint once we add it
        public override Vector3[] GetJointPositions(Handedness hand) => _positions;
        public override Quaternion[] GetJointRotations(Handedness hand) => new Quaternion[21];
    }
}
