import React from "react";
import axios from "axios";
import "./ModelSelect.css";

const API_URL = process.env.REACT_APP_API_URL;

const ModelSelect = ({ selectedModel, setSelectedModel }) => {
    const models = [
        { label: "Faster R-CNN", value: "pretrained_faster_rcnn" },
        { label: "Vision Transformer (ViT)", value: "ViT_model" },
    ];

    const handleModelChange = async (modelValue) => {
        try {
        const response = await axios.post(
            `${API_URL}/change_model`,
            { model: modelValue },  // Send Base64 image
            { headers: { "Content-Type": "application/json" } }
        );

        setSelectedModel(response.data.model);
        } catch (error) {
        console.error("Error changing model:", error);
        }
    };


    return (
        <div>
        <h2>Select Model:</h2>
        <div>
            {models.map(({ label, value }) => (
            <button
                key={value}
                className={`model-button ${selectedModel === value ? "active" : ""}`}
                onClick={() => handleModelChange(value)}
            >
                {label}
            </button>
            ))}
        </div>
        </div>
    );
    
    };

export default ModelSelect;
