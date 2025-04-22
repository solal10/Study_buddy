import PropTypes from 'prop-types';
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const TopicItem = ({ topic, completed, onToggle, onAskTopic }) => {
  const navigate = useNavigate();
  return (
  <div className="flex items-center justify-between py-2 pl-4">
    <div className="flex items-center">
      <input
        type="checkbox"
        checked={completed}
        onChange={() => onToggle(topic)}
        className="h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
      />
      <span className="ml-3 text-gray-700">
        {topic.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
      </span>
    </div>
    <div className="space-x-2">
      <button
        onClick={() => onAskTopic(topic)}
        className="px-3 py-1 text-sm text-blue-600 hover:bg-blue-50 rounded-md transition-colors"
      >
        Ask Study Buddy
      </button>
      <button
        onClick={() => navigate(`/quiz/${topic}`)}
        className="px-3 py-1 text-sm text-green-600 hover:bg-green-50 rounded-md transition-colors"
      >
        Take Quiz
      </button>
    </div>
  </div>
  );
};

TopicItem.propTypes = {
  topic: PropTypes.string.isRequired,
  completed: PropTypes.bool.isRequired,
  onToggle: PropTypes.func.isRequired,
  onAskTopic: PropTypes.func.isRequired,
};

const CategoryDropdown = ({ title, topics, completedTopics, onToggle, onAskTopic }) => {
  const [isOpen, setIsOpen] = useState(false);
  const completedCount = topics.filter(topic => completedTopics.includes(topic)).length;

  return (
    <div className="mb-4">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-3 bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors"
      >
        <div className="flex items-center">
          <span className="font-medium text-gray-900">{title}</span>
          <span className="ml-2 text-sm text-gray-500">
            ({completedCount}/{topics.length})
          </span>
        </div>
        <svg
          className={`w-5 h-5 transition-transform ${isOpen ? 'transform rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {isOpen && (
        <div className="mt-2 space-y-1 border-l-2 border-gray-200">
          {topics.map((topic) => (
            <TopicItem
              key={topic}
              topic={topic}
              completed={completedTopics.includes(topic)}
              onToggle={onToggle}
              onAskTopic={onAskTopic}
            />
          ))}
        </div>
      )}
    </div>
  );
};

CategoryDropdown.propTypes = {
  title: PropTypes.string.isRequired,
  topics: PropTypes.arrayOf(PropTypes.string).isRequired,
  completedTopics: PropTypes.arrayOf(PropTypes.string).isRequired,
  onToggle: PropTypes.func.isRequired,
  onAskTopic: PropTypes.func.isRequired,
};

const Sidebar = ({ sessionId, onAskTopic }) => {
  const [completedTopics, setCompletedTopics] = useState([]);
  const [loading, setLoading] = useState(true);

  const topics = {
    '📚 Foundations': [
      'what-is-ai',
      'ai-applications',
      'ethics-in-ai'
    ],
    '🧠 Machine Learning': [
      'machine-learning-basics',
      'supervised-learning',
      'unsupervised-learning',
      'reinforcement-learning'
    ],
    '🧮 Neural Networks & Vision': [
      'neural-networks',
      'computer-vision'
    ],
    '💬 Language Processing': [
      'natural-language-processing'
    ],
    '🛠 ML Workflow': [
      'data-preprocessing',
      'model-evaluation'
    ]
  };

  useEffect(() => {
    const fetchTopics = async () => {
      try {
        const response = await fetch(`http://localhost:5001/topics-status?session_id=${sessionId}`);
        if (!response.ok) throw new Error('Failed to fetch topics');
        const data = await response.json();
        setCompletedTopics(data.completed_topics || []);
      } catch (error) {
        console.error('Error fetching topics:', error);
      } finally {
        setLoading(false);
      }
    };

    if (sessionId) {
      fetchTopics();
    } else {
      setLoading(false);
    }
  }, [sessionId]);

  const handleTopicToggle = async (topic) => {
    try {
      // Optimistic update
      const newCompletedTopics = completedTopics.includes(topic)
        ? completedTopics.filter(t => t !== topic)
        : [...completedTopics, topic];
      
      setCompletedTopics(newCompletedTopics);

      const response = await fetch('http://localhost:5001/mark_topic', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          topic,
          completed: !completedTopics.includes(topic),
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to update topic');
      }
    } catch (error) {
      console.error('Error updating topic:', error);
      // Revert the optimistic update on error
      setCompletedTopics(completedTopics);
    }
  };

  if (loading) {
    return <div className="p-4">Loading...</div>;
  }

  const totalTopics = Object.values(topics).flat().length;
  const completionPercentage = (completedTopics.length / totalTopics) * 100;

  return (
    <div className="w-64 bg-white rounded-lg shadow-lg p-4">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-lg font-semibold text-gray-900">Progress</h2>
          <span className="text-sm text-gray-500">{Math.round(completionPercentage)}%</span>
        </div>
        <div className="h-2 bg-gray-200 rounded-full">
          <div
            className="h-2 bg-blue-600 rounded-full transition-all duration-500"
            style={{ width: `${completionPercentage}%` }}
          />
        </div>
      </div>

      <div className="space-y-2">
        {Object.entries(topics).map(([category, topicList]) => (
          <CategoryDropdown
            key={category}
            title={category}
            topics={topicList}
            completedTopics={completedTopics}
            onToggle={handleTopicToggle}
            onAskTopic={onAskTopic}
          />
        ))}
      </div>
    </div>
  );
};

Sidebar.propTypes = {
  sessionId: PropTypes.string,
  onAskTopic: PropTypes.func.isRequired,
};

export default Sidebar;
