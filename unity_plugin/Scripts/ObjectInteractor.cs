using UnityEngine;

namespace OpenGestureXR
{
    /// <summary>
    /// Attach to any GameObject you want to be gesture-interactable.
    /// Listens for gesture events and maps them to actions.
    /// </summary>
    public class ObjectInteractor : MonoBehaviour
    {
        public Transform handAnchor;

        [Range(0f, 1f)]
        public float confidenceThreshold = 0.7f;

        bool _grabbed;
        string _prevGesture;
        int _sameCount;
        Renderer _rend;
        Color _origColor;

        // require a few consecutive frames of the same gesture before acting,
        // otherwise you get tons of flickering between states
        const int kSmoothFrames = 3;

        void Awake()
        {
            _rend = GetComponent<Renderer>();
            if (_rend) _origColor = _rend.material.color;
        }

        void OnEnable()  => GestureClient.OnGestureReceived += OnGesture;
        void OnDisable() => GestureClient.OnGestureReceived -= OnGesture;

        void OnGesture(string gesture, float confidence)
        {
            if (confidence < confidenceThreshold) return;

            // basic temporal smoothing
            if (gesture == _prevGesture)
                _sameCount++;
            else
            {
                _prevGesture = gesture;
                _sameCount = 1;
            }
            if (_sameCount < kSmoothFrames) return;

            switch (gesture)
            {
                case "grab":       DoGrab();      break;
                case "open_hand":  DoRelease();   break;
                case "pinch":      DoSelect();    break;
                case "point":      DoHighlight(); break;
                case "thumbs_up":  DoConfirm();   break;
                case "peace":      DoReset();     break;
            }
        }

        void DoGrab()
        {
            if (_grabbed || !handAnchor) return;
            _grabbed = true;
            transform.SetParent(handAnchor, true);
            Debug.Log($"grabbed {name}");
        }

        void DoRelease()
        {
            if (!_grabbed) return;
            _grabbed = false;
            transform.SetParent(null);
        }

        void DoSelect()
        {
            if (_rend) _rend.material.color = Color.cyan;
        }

        void DoHighlight()
        {
            if (_rend) _rend.material.color = Color.yellow;
        }

        void DoConfirm()
        {
            // TODO: hook this up to a proper UI confirmation flow
            if (_rend) _rend.material.color = Color.green;
        }

        void DoReset()
        {
            if (_rend) _rend.material.color = _origColor;
        }
    }
}
