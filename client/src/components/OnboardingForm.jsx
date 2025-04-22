import { useState } from 'react'

export default function OnboardingForm({ onComplete }) {
  const [formData, setFormData] = useState({
    knowledge_level: '',
    goals: '',
    pace: '',
    interests: []
  })

  const handleSubmit = async (e) => {
    e.preventDefault()

    // Validate form
    if (!formData.knowledge_level || !formData.goals || !formData.pace) {
      alert('Please fill in all required fields')
      return
    }

    const token = localStorage.getItem('token')
    if (!token) {
      alert('Please log in first')
      return
    }

    try {
      const response = await fetch('http://localhost:5001/onboarding', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(formData)
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.error || 'Network response was not ok')
      }

      const data = await response.json()
      onComplete()
    } catch (error) {
      console.error('Error:', error)
      alert(error.message || 'Failed to save preferences. Please try again.')
    }
  }

  return (
    <div className="max-w-2xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold text-center text-gray-800 mb-8">Welcome to AI Study Buddy!</h2>
      
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Knowledge Level */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">Your current knowledge level in AI/ML:</label>
          <select
            value={formData.knowledge_level}
            onChange={(e) => setFormData(prev => ({ ...prev, knowledge_level: e.target.value }))}
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
            required
          >
            <option value="">Select your level</option>
            <option value="beginner">Beginner - New to AI/ML</option>
            <option value="intermediate">Intermediate - Some experience</option>
            <option value="advanced">Advanced - Significant experience</option>
          </select>
        </div>

        {/* Learning Goals */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">What are your learning goals?</label>
          <textarea
            value={formData.goals}
            onChange={(e) => setFormData(prev => ({ ...prev, goals: e.target.value }))}
            className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
            rows="3"
            placeholder="E.g., Understanding AI fundamentals, building ML models, etc."
            required
          />
        </div>

        {/* Learning Pace */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">Preferred learning pace:</label>
          <select
            value={formData.pace}
            onChange={(e) => setFormData(prev => ({ ...prev, pace: e.target.value }))}
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
            required
          >
            <option value="">Select your pace</option>
            <option value="relaxed">Relaxed - 1-2 hours per week</option>
            <option value="moderate">Moderate - 3-5 hours per week</option>
            <option value="intensive">Intensive - 6+ hours per week</option>
          </select>
        </div>

        {/* Specific Interests */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">Areas of interest (optional):</label>
          <div className="grid grid-cols-2 gap-4">
            {[
              'Deep Learning',
              'Computer Vision',
              'Natural Language Processing',
              'Reinforcement Learning',
              'Ethics in AI',
              'AI Applications',
              'Machine Learning Theory',
              'Data Science'
            ].map(interest => (
              <label key={interest} className="inline-flex items-center">
                <input
                  type="checkbox"
                  value={interest}
                  checked={formData.interests.includes(interest)}
                  onChange={(e) => {
                    const isChecked = e.target.checked
                    setFormData(prev => ({
                      ...prev,
                      interests: isChecked
                        ? [...prev.interests, interest]
                        : prev.interests.filter(i => i !== interest)
                    }))
                  }}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <span className="ml-2 text-sm text-gray-600">{interest}</span>
              </label>
            ))}
          </div>
        </div>

        <div className="flex justify-end">
          <button
            type="submit"
            className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            Start Learning
          </button>
        </div>
      </form>
    </div>
  )
}
