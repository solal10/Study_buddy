import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { quizData } from '../data/quizData';

const QuizPage = () => {
  const { topic } = useParams();
  const [userAnswers, setUserAnswers] = useState({});
  const [showResults, setShowResults] = useState(false);
  const [sessionId, setSessionId] = useState(null);

  useEffect(() => {
    // Get session ID from localStorage
    const storedSessionId = localStorage.getItem('session_id');
    setSessionId(storedSessionId);
  }, []);

  // Get quiz data for this topic
  console.log('Topic:', topic);
  console.log('Quiz data:', quizData);
  const quiz = quizData[topic];
  console.log('Quiz for topic:', quiz);
  if (!quiz) {
    return (
      <div className="min-h-screen bg-gray-100 p-8">
        <div className="max-w-2xl mx-auto bg-white rounded-lg shadow-lg p-6">
          <h1 className="text-2xl font-bold text-red-600 mb-4">Quiz Not Found</h1>
          <p className="text-gray-600 mb-4">
            Sorry, we couldn't find a quiz for this topic. Please try another topic.
          </p>
          <Link
            to="/"
            className="inline-block px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
          >
            Back to Chat
          </Link>
        </div>
      </div>
    );
  }

  // Calculate score after submission
  const calculateScore = () => {
    let correct = 0;
    quiz.questions.forEach((q, index) => {
      if (userAnswers[index] === q.correctAnswer) {
        correct++;
      }
    });
    return {
      correct,
      total: quiz.questions.length,
      percentage: Math.round((correct / quiz.questions.length) * 100)
    };
  };

  // Handle answer selection
  const handleAnswerSelect = (questionIndex, answer) => {
    if (!showResults) {
      setUserAnswers(prev => ({
        ...prev,
        [questionIndex]: answer
      }));
    }
  };

  // Handle quiz submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    const score = calculateScore();
    setShowResults(true);

    try {
      const response = await fetch('http://localhost:5001/submit-quiz', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          topic: topic,
          score: score.correct,
          total: score.total,
          timestamp: new Date().toISOString()
        })
      });

      if (!response.ok) {
        console.error('Failed to save quiz results');
      }
    } catch (error) {
      console.error('Error saving quiz results:', error);
    }
  };

  // Get result status for a question
  const getQuestionResult = (questionIndex) => {
    if (!showResults) return null;
    const isCorrect = userAnswers[questionIndex] === quiz.questions[questionIndex].correctAnswer;
    return isCorrect ? '✅' : '❌';
  };

  const score = showResults ? calculateScore() : null;

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-2xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex justify-between items-center mb-6">
            <h1 className="text-2xl font-bold text-gray-900">{quiz.title} Quiz</h1>
            <Link
              to="/"
              className="px-4 py-2 text-sm text-blue-600 hover:bg-blue-50 rounded-md transition-colors"
            >
              Back to Chat
            </Link>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="space-y-6">
              {quiz.questions.map((q, questionIndex) => (
                <div key={questionIndex} className="p-4 border rounded-lg">
                  <div className="flex justify-between items-start mb-3">
                    <p className="text-lg font-medium">
                      {questionIndex + 1}. {q.question}
                    </p>
                    {getQuestionResult(questionIndex) && (
                      <span className="text-xl">{getQuestionResult(questionIndex)}</span>
                    )}
                  </div>

                  <div className="space-y-2">
                    {q.options.map((option, optionIndex) => (
                      <label
                        key={optionIndex}
                        className={`
                          block p-3 rounded-lg border cursor-pointer transition-colors
                          ${
                            userAnswers[questionIndex] === option
                              ? 'bg-blue-50 border-blue-200'
                              : 'hover:bg-gray-50'
                          }
                          ${
                            showResults &&
                            option === q.correctAnswer &&
                            'bg-green-50 border-green-200'
                          }
                          ${
                            showResults &&
                            userAnswers[questionIndex] === option &&
                            option !== q.correctAnswer &&
                            'bg-red-50 border-red-200'
                          }
                        `}
                      >
                        <div className="flex items-center">
                          <input
                            type="radio"
                            name={`question-${questionIndex}`}
                            value={option}
                            checked={userAnswers[questionIndex] === option}
                            onChange={() => handleAnswerSelect(questionIndex, option)}
                            disabled={showResults}
                            className="h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500"
                          />
                          <span className="ml-3">{option}</span>
                        </div>
                      </label>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {!showResults && (
              <div className="mt-6">
                <button
                  type="submit"
                  className="w-full px-4 py-2 text-white bg-blue-600 rounded-md hover:bg-blue-700 transition-colors"
                >
                  Submit Quiz
                </button>
              </div>
            )}
          </form>

          {showResults && (
            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
              <h2 className="text-xl font-semibold mb-2">Quiz Results</h2>
              <p className="text-lg">
                Score: {score.correct} out of {score.total} ({score.percentage}%)
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default QuizPage;
