import { useState } from "react";
import "./Assistant.css";

interface ApiResponse {
  answer: string;
}

interface ApiRequest {
  question: string;
  max_length: number;
}

export default function MentalHealthAssistant() {
  const [input, setInput] = useState<string>("");
  const [suggestion, setSuggestion] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  const generateSuggestion = async (): Promise<void> => {
    if (!input.trim()) {
      setError(
        "Please describe the challenge you're facing with your patient."
      );
      return;
    }

    setLoading(true);
    setError("");
    setSuggestion("");

    try {
      // Call FastAPI server
      const requestBody: ApiRequest = {
        question: input,
        max_length: 200,
      };

      const response = await fetch(
        "https://legacy-takehome-production.up.railway.app/api/generate-advice",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "Failed to generate suggestion");
      }

      const data: ApiResponse = await response.json();
      setSuggestion(data.answer);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError(
          "There was an error generating a suggestion. Please try again."
        );
      }
      console.error("Error:", err);
    } finally {
      setLoading(false);
    }
  };

  const clearForm = (): void => {
    setInput("");
    setSuggestion("");
    setError("");
  };

  return (
    <div className="container">
      <div className="main-content">
        {/* Header */}
        <div className="header">
          <h1 className="title">Mental Health Counseling Assistant</h1>
        </div>

        {/* Main Form */}
        <div className="form-card">
          <div className="input-section">
            <label htmlFor="challenge" className="label">
              Describe your patient challenge 🥼📋:
            </label>
            <textarea
              id="challenge"
              value={input}
              onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) =>
                setInput(e.target.value)
              }
              placeholder="Example: My patient has been feeling anxious. How can I help them?"
              className="textarea"
              disabled={loading}
            />
            <p className="help-text">
              TIP: Try describing the challenge as a question from the
              perspective of the patient.
            </p>
          </div>

          {/* Error Message */}
          {error && (
            <div className="error-message">
              <p>{error}</p>
            </div>
          )}

          {/* Action Buttons */}
          <div className="button-group">
            <button
              onClick={generateSuggestion}
              disabled={loading || !input.trim()}
              className={`primary-button ${
                loading || !input.trim() ? "disabled" : ""
              }`}
            >
              {loading ? (
                <span className="loading-content">
                  <span className="spinner"></span>
                  Generating response...
                </span>
              ) : (
                "Get suggestion"
              )}
            </button>

            <button
              onClick={clearForm}
              disabled={loading}
              className={`secondary-button ${loading ? "disabled" : ""}`}
            >
              Clear
            </button>
          </div>

          {/* Generated Suggestion */}
          {suggestion && (
            <div className="results-section">
              <h3 className="results-title">Suggested Approach:</h3>
              <div className="suggestion-box">
                <p className="suggestion-text">{suggestion}</p>
              </div>
              <div className="disclaimer">
                <p>
                  This suggestion is AI generated. Please double check
                  responses.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
