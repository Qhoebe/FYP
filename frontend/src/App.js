import React, { useState, useRef } from "react";
import axios from "axios";
import CameraView from "./components/CameraView";
import ModelResult from "./components/ModelResult";
import ModelSelect from "./components/ModelSelect";
import "./App.css";

const API_URL = process.env.REACT_APP_API_URL;

const App = () => {
    const [count, setCount] = useState(null);
    const [isRunning, setIsRunning] = useState(false);
    const isRunningRef = useRef(false); // Create a ref for isRunning
    const cameraRef = useRef(null);
    const intervalRef = useRef(null);
    const [model, setModel] = useState("no model selected");

    const sendToModel = async () => {
        if (!cameraRef.current) return;
        const imageData = cameraRef.current.captureFrame();
        if (!imageData) return;
    
        try {
            const response = await axios.post(
                `${API_URL}/evaluate`,
                { image: imageData },
                { headers: { "Content-Type": "application/json" } }
            );

            if (isRunningRef.current) {  // Use the ref instead of stale state
                setCount(response.data.count);
            }
            
        } catch (error) {
            console.error("Error sending image to model:", error);
        }
    };

    const startEvaluation = () => {
        if (isRunningRef.current) return; // Prevent multiple intervals
        setIsRunning(true);
        isRunningRef.current = true;  // Update the ref
        intervalRef.current = setInterval(sendToModel, 1000);
    };

    const stopEvaluation = () => {
        setIsRunning(false);
        isRunningRef.current = false;  //  Update the ref
        setCount(null);
        clearInterval(intervalRef.current);
    };

    return (
        <div className="app-container">
            <h1 className="header">Smart Object Counter</h1>
            <div className="main-container">
                <div className="left-section">
                    <CameraView ref={cameraRef} />
                    <button className="control-button" onClick={isRunning ? stopEvaluation : startEvaluation}>
                        {isRunning ? "Stop" : "Start"}
                    </button>
                </div>
                <div className="right-section">
                    <ModelSelect selectedModel={model} setSelectedModel={setModel} />
                    <div className="same-line">
                        <h2>Current Model:</h2>
                        <p>{model}</p>
                    </div>
                    <ModelResult count={count} />
                </div>
            </div>
        </div>
    );
};

export default App;
