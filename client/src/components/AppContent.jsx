import { useState, useRef, useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import OnboardingForm from './OnboardingForm'
import Sidebar from './Sidebar'
import QuizPage from './QuizPage'
import Login from './Login'
import Register from './Register'
import PrivateRoute from './PrivateRoute'

function AppContent() {
  const { user } = useAuth()
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showOnboarding, setShowOnboarding] = useState(false)
  const [topics, setTopics] = useState([])
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Fetch initial message when component mounts
  useEffect(() => {
    if (!user) return

    const fetchInitialMessage = async () => {
      try {
        const token = localStorage.getItem('token')
        // First check if user needs onboarding
        const checkResponse = await fetch('http://localhost:5001/ask', {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          body: JSON.stringify({ 
            message: ''
          })
        })
        
        if (!checkResponse.ok) {
          throw new Error('Network response was not ok')
        }

        const checkData = await checkResponse.json()
        
        if (checkData.response === 'Please complete the onboarding form first.') {
          setShowOnboarding(true)
          return
        }

        // User is onboarded, show welcome message
        const welcomeResponse = await fetch('http://localhost:5001/welcome-message', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        })
        if (!welcomeResponse.ok) {
          throw new Error('Network response was not ok')
        }

        const welcomeData = await welcomeResponse.json()
        setMessages([{ role: 'assistant', content: welcomeData.message }])
      } catch (error) {
        console.error('Error:', error)
        if (error.message.includes('Network response was not ok')) {
          localStorage.removeItem('token')
          navigate('/login')
        }
      }
    }

    fetchInitialMessage()
  }, [navigate])

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim()) return

    const userMessage = input.trim()
    setIsLoading(true)

    try {
      const token = localStorage.getItem('token')
      if (!token) {
        navigate('/login')
        return
      }

      // Add user message to chat
      const newMessage = { role: 'user', content: userMessage }
      setMessages(prev => [...prev, newMessage])
      setInput('')

      const response = await fetch('http://localhost:5001/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          message: userMessage
        })
      })

      const data = await response.json()
      
      if (!response.ok) {
        throw new Error(data.error || 'Network response was not ok')
      }

      if (data.response === 'Please complete the onboarding form first.') {
        setShowOnboarding(true)
        return
      }

      // Add assistant's response to chat
      const assistantMessage = { role: 'assistant', content: data.response }
      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Error:', error)
      const errorMessage = { 
        role: 'assistant', 
        content: `Sorry, I encountered an error: ${error.message}. Please try again.` 
      }
      setMessages(prev => [...prev, errorMessage])
      if (error.message.includes('Network response was not ok')) {
        localStorage.removeItem('token')
        navigate('/login')
      }
    } finally {
      setIsLoading(false)
    }
  }

  const LoadingIndicator = () => (
    <div className="flex justify-center">
      <div className="animate-pulse text-gray-400">Thinking...</div>
    </div>
  )

  const handleOnboardingComplete = async () => {
    try {
      const token = localStorage.getItem('token')
      if (!token) {
        navigate('/login')
        return
      }

      const response = await fetch('http://localhost:5001/welcome-message', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      
      if (!response.ok) throw new Error('Network response was not ok')
      
      const data = await response.json()
      setMessages([{ role: 'assistant', content: data.message }])
      setShowOnboarding(false)
    } catch (error) {
      console.error('Error fetching welcome message:', error)
      if (error.message.includes('Network response was not ok')) {
        localStorage.removeItem('token')
        navigate('/login')
      }
    }
  }

  // Handle topic selection from sidebar
  const handleTopicClick = (topic) => {
    setInput(`Tell me about ${topic}`)
    setMessages([])
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route
          path="/quiz"
          element={
            <PrivateRoute>
              <QuizPage />
            </PrivateRoute>
          }
        />
        <Route
          path="/"
          element={
            <PrivateRoute>
              <div className="flex h-screen">
                <Sidebar topics={topics} onTopicClick={handleTopicClick} />
                <main className="flex-1 p-6 overflow-auto">
                  {showOnboarding ? (
                    <OnboardingForm onComplete={() => setShowOnboarding(false)} />
                  ) : (
                    <div className="max-w-4xl mx-auto space-y-4">
                      <div className="space-y-4 pb-[100px]">
                        {messages.map((msg, index) => (
                          <div
                            key={index}
                            className={`p-4 rounded-lg ${
                              msg.role === 'user'
                                ? 'bg-blue-100 ml-auto max-w-[80%]'
                                : 'bg-white max-w-[80%]'
                            }`}
                          >
                            <p className="whitespace-pre-wrap">{msg.content}</p>
                          </div>
                        ))}
                        {isLoading && (
                          <div className="flex items-center space-x-2">
                            <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" />
                            <div
                              className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
                              style={{ animationDelay: '0.2s' }}
                            />
                            <div
                              className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
                              style={{ animationDelay: '0.4s' }}
                            />
                          </div>
                        )}
                        <div ref={messagesEndRef} />
                      </div>

                      <div className="fixed bottom-0 left-0 right-0 bg-white p-4 border-t">
                        <div className="max-w-4xl mx-auto">
                          <form onSubmit={handleSubmit} className="flex gap-4">
                            <input
                              type="text"
                              value={input}
                              onChange={(e) => setInput(e.target.value)}
                              placeholder="Ask me anything about AI..."
                              className="flex-1 p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                            />
                            <button
                              type="submit"
                              disabled={isLoading}
                              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50"
                            >
                              Send
                            </button>
                          </form>
                        </div>
                      </div>
                    </div>
                  )}
                </main>
              </div>
            </PrivateRoute>
          }
        />
      </Routes>
    </div>
  )
}

export default AppContent
