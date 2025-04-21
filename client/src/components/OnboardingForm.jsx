import { useState, useEffect } from 'react'
import { v4 as uuidv4 } from 'uuid'

export default function OnboardingForm({ onComplete }) {
  const [sessionId, setSessionId] = useState('')
  const [formData, setFormData] = useState({
    level: '',
    preference: ''
  })

  useEffect(() => {
    // Get existing session ID or create new one
    const existingSessionId = localStorage.getItem('sessionId')
    if (existingSessionId) {
      setSessionId(existingSessionId)
    } else {
      const newSessionId = uuidv4()
      localStorage.setItem('sessionId', newSessionId)
      setSessionId(newSessionId)
    }
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()

    // Validate form
    if (!formData.level || !formData.preference) {
      alert('Please select your knowledge level and learning preference')
      return
    }

    // Generate a unique session ID
    const sessionId = uuidv4()
    localStorage.setItem('sessionId', sessionId)

    try {
      const response = await fetch('http://localhost:5001/onboarding', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          session_id: sessionId,
          level: formData.level,
          preference: formData.preference
        })
      })

      if (!response.ok) {
        throw new Error('Network response was not ok')
      }

      const data = await response.json()
      
      if (data.success) {
        onComplete()
      } else {
        throw new Error(data.error || 'Failed to save preferences')
      }
    } catch (error) {
      console.error('Error:', error)
      alert('Failed to save preferences. Please try again.')
      localStorage.removeItem('sessionId')
    }
  }

  const handleLevelChange = (e) => {
    setFormData(prev => ({ ...prev, level: e.target.value }))
  }

  const handlePreferenceChange = (e) => {
    setFormData(prev => ({ ...prev, preference: e.target.value }))
  }

  const handleInterestChange = (e) => {
    const value = e.target.value
    if (value === 'Other') {
      setFormData(prev => ({
        ...prev,
        showOtherInput: e.target.checked,
        otherInterest: e.target.checked ? prev.otherInterest : '',
        interests: e.target.checked 
          ? prev.interests
          : prev.interests.filter(i => !i.startsWith('Other:'))
      }))
    } else {
      setFormData(prev => ({
        ...prev,
        interests: e.target.checked
          ? [...prev.interests, value]
          : prev.interests.filter(i => i !== value)
      }))
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6 p-6">
      <h2 className="text-2xl font-bold text-center text-gray-800 mb-8">Welcome to AI Study Buddy!</h2>
      
      {/* Knowledge Level */}
      <div className="space-y-4">
        <label className="block text-lg font-medium text-gray-700">Your knowledge level:</label>
        <div className="space-y-2">
          {['beginner', 'intermediate', 'advanced'].map((level) => (
            <label key={level} className="flex items-center p-3 border rounded-lg hover:bg-gray-50 cursor-pointer">
              <input
                type="radio"
                name="level"
                value={level}
                checked={formData.level === level}
                onChange={handleLevelChange}
                className="form-radio h-4 w-4 text-blue-600"
              />
              <span className="ml-2 capitalize">{level}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Learning Preference */}
      <div className="space-y-4">
        <label className="block text-lg font-medium text-gray-700">How do you prefer to learn?</label>
        <div className="space-y-2">
          {[
            'short explanations',
            'hands-on code',
            'in-depth theory'
          ].map((pref) => (
            <label key={pref} className="flex items-center p-3 border rounded-lg hover:bg-gray-50 cursor-pointer">
              <input
                type="radio"
                name="preference"
                value={pref}
                checked={formData.preference === pref}
                onChange={handlePreferenceChange}
                className="form-radio h-4 w-4 text-blue-600"
              />
              <span className="ml-2 capitalize">{pref}</span>
            </label>
          ))}
        </div>
      </div>

      <button
        type="submit"
        className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors"
      >
        Start Learning
      </button>
    </form>
  )
}
