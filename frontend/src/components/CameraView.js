import React, { useRef, useEffect, useImperativeHandle, forwardRef } from "react";

const CameraView = forwardRef((props, ref) => {
    const videoRef = useRef(null);

    useEffect(() => {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                videoRef.current.srcObject = stream;
            })
            .catch(err => console.error("Error accessing camera:", err));
    }, []);

    const captureFrame = () => {
        if (!videoRef.current) return null;

        const canvas = document.createElement("canvas");
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL("image/jpeg");
    };

    useImperativeHandle(ref, () => ({
        captureFrame,
    }));

    return <video ref={videoRef} autoPlay playsInline />;
});

export default CameraView;
