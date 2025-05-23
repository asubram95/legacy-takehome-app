import { useState } from 'react';
import './Assistant.css';

interface ApiResponse {
  answer: string;
}

interface ApiRequest {
  question: string;
  max_length: number;
}

export default function MentalHealthAssistant() {
  const [challenge, setChallenge] = useState<string>('');
  const [suggestion, setSuggestion] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const generateSuggestion = async (): Promise<void> => {
    if (!challenge.trim()) {
      setError('Please describe the challenge you\'re facing with your patient.');
      return;
    }

    setLoading(true);
    setError('');
    setSuggestion('');

    try {
      // This will call your Python API when it's set up
      const requestBody: ApiRequest = { 
        question: challenge,
        max_length: 200 
      };

      const response = await fetch('/api/generate-advice', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error('Failed to generate suggestion');
      }

      const data: ApiResponse = await response.json();
      setSuggestion(data.answer);
    } catch (err) {
      setError('There was an error generating a suggestion. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const clearForm = (): void => {
    setChallenge('');
    setSuggestion('');
    setError('');
  };

  return (
    <div className="container">
      <div className="main-content">
        {/* Header */}
        <div className="header">
          <h1 className="title">
            Mental Health Counseling Assistant
          </h1>
        </div>

        {/* Main Form */}
        <div className="form-card">
          <div className="input-section">
            <label htmlFor="challenge" className="label">
              Describe your patient challenge:
            </label>
            <textarea
              id="challenge"
              value={challenge}
              onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setChallenge(e.target.value)}
              placeholder="Example: My patient has been showing signs of increased anxiety during our sessions. They seem reluctant to discuss their feelings and often deflect when I try to explore deeper issues. How can I help them feel more comfortable opening up?"
              className="textarea"
              disabled={loading}
            />
            <p className="help-text">
              TIP: Try describing the challenge as a question from the perspective of the patient.
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
              disabled={loading || !challenge.trim()}
              className={`primary-button ${loading || !challenge.trim() ? 'disabled' : ''}`}
            >
              {loading ? (
                <span className="loading-content">
                  <span className="spinner"></span>
                  Generating Guidance...
                </span>
              ) : (
                'Get suggestion'
              )}
            </button>
            
            <button
              onClick={clearForm}
              disabled={loading}
              className={`secondary-button ${loading ? 'disabled' : ''}`}
            >
              Clear
            </button>
          </div>

          {/* Generated Suggestion */}
          {suggestion && (
            <div className="results-section">
              <h3 className="results-title">
                Suggested Approach:
              </h3>
              <div className="suggestion-box">
                <p className="suggestion-text">
                  {suggestion}
                </p>
              </div>
              <div className="disclaimer">
                <p>
                  <strong>Important:</strong> This guidance is AI-generated and should be considered as one perspective among many. 
                  Always use your professional judgment and consider consulting with colleagues or supervisors when appropriate.
                </p>
              </div>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}