import React, { useState, useRef } from "react";
import axios from "axios";
import CameraView from "./components/CameraView";
import ModelResult from "./components/ModelResult";
import "./App.css";

const API_URL = process.env.REACT_APP_API_URL;

const App = () => {
    const [count, setCount] = useState(null);
    const [isRunning, setIsRunning] = useState(false);
    const cameraRef = useRef(null);
    const intervalRef = useRef(null);

    const sendToModel = async () => {
      if (!cameraRef.current) return;
      const imageData = cameraRef.current.captureFrame(); // Base64 image string
      if (!imageData) return;
  
      try {
          const response = await axios.post(
              `${API_URL}/evaluate/model_1`,
              { image: imageData },  // Send Base64 image
              { headers: { "Content-Type": "application/json" } }
          );
          setCount(response.data.count);
      } catch (error) {
          console.error("Error sending image to model:", error);
      }
  };
  

    const startEvaluation = () => {
        if (isRunning) return;
        setIsRunning(true);
        intervalRef.current = setInterval(sendToModel, 2000);
    };

    const stopEvaluation = () => {
        setIsRunning(false);
        setCount(null);
        clearInterval(intervalRef.current);
    };

    return (
        <div className="app-container">
            {/* Centered Header */}
            <h1 className="header">Smart Object Counter</h1>

            {/* Main Container with Two Equal Sections */}
            <div className="main-container">
                {/* Left Section (Camera + Button) */}
                <div className="left-section">
                    <CameraView ref={cameraRef} />
                    <button className="control-button" onClick={isRunning ? stopEvaluation : startEvaluation}>
                        {isRunning ? "Stop" : "Start"}
                    </button>
                </div>

                {/* Right Section (Model Results) */}
                <div className="right-section">
                    <ModelResult count={count} />
                </div>
            </div>
        </div>
    );
};

export default App;
