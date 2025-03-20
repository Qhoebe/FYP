import React from "react";
import "./ModelResult.css";

const ModelResult = ({ count }) => {
    return (
        <div className="same-line">
            <h2>Object Count:</h2>
            <p>{count !== null ? count : "Waiting for results..."}</p>
        </div>
    );
};

export default ModelResult;
