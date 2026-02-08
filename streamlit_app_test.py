#!/usr/bin/env python3
# minimal test - just show camera feed

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from av import VideoFrame

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class SimpleVideoProcessor(VideoProcessorBase):
    def recv(self, frame: VideoFrame) -> VideoFrame:
        return frame

def main():
    st.set_page_config(
        page_title="Camera Test",
        layout="wide"
    )
    
    st.title("Camera Feed Test - No Processing")
    st.markdown("This is a minimal test to see if webrtc works without any face recognition or MediaPipe processing.")
    
    ctx = webrtc_streamer(
        key="camera-test",
        video_processor_factory=SimpleVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 30}
            },
            "audio": False
        },
        async_processing=False,
    )
    
    if ctx.video_processor:
        st.success("Camera stream is active!")
    else:
        st.info("Click START to begin camera stream")

if __name__ == "__main__":
    main()

