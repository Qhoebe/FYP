import React from "react";

const ModelResult = ({ count }) => {
    return (
        <div>
            <h2>Object Count:</h2>
            <p>{count !== null ? count : "Waiting for results..."}</p>
        </div>
    );
};

export default ModelResult;
