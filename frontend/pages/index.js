import { useState } from 'react';
import axios from 'axios';

export default function Home() {
  const [inputText, setInputText] = useState('');
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [stats, setStats] = useState(null);
  const [summaryLanguage, setSummaryLanguage] = useState('tamil');
  const [summaryEnglish, setSummaryEnglish] = useState('');
  const [translating, setTranslating] = useState(false);
  const [playingAudio, setPlayingAudio] = useState(null);
  const [chatQuestion, setChatQuestion] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [useTransliteration, setUseTransliteration] = useState(true);
  const [previewTamil, setPreviewTamil] = useState('');
  const [useMyMemory, setUseMyMemory] = useState(true);
  const [translationMethod, setTranslationMethod] = useState('');


  const handleSummarize = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text to summarize');
      return;
    }

    setLoading(true);
    setError('');
    setSummary('');
    setStats(null);
    setSummaryLanguage('tamil');
    setSummaryEnglish('');

    try {
      const res = await axios.post('http://127.0.0.1:8000/summarize', {
        text: inputText,
        max_length: 150,
        min_length: 40
      });
      setInputText(res.data.original);
      
      if (res.data.summary_english) {
        setSummaryEnglish(res.data.summary_english);
      }
      
      setSummary(res.data.summary);
      setStats({
        originalLength: res.data.stats.original_length,
        summaryLength: res.data.stats.summary_length,
        compressionRatio: res.data.stats.compression_ratio,
        mode: res.data.mode
      });
      
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || 'Error summarizing. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleTranslate = async () => {
    if (!summary.trim()) {
      setError('Please summarize text first before translating');
      return;
    }

    // If currently showing English, switch back to Tamil
    if (summaryLanguage === 'english') {
      setSummaryLanguage('tamil');
      return;
    }

    // If we have English translation cached, switch to English
    if (summaryEnglish && summaryEnglish.trim()) {
      setSummaryLanguage('english');
      return;
    }

    // Otherwise, fetch English translation
    setTranslating(true);
    setError('');

    try {
      const res = await axios.post('http://127.0.0.1:8000/translate-summary', {
        summary_text: summary,
        use_mymemory: useMyMemory
      });
      
      setSummaryEnglish(res.data.summary_english);
      setTranslationMethod(res.data.translation_method);
      setSummaryLanguage('english');
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || 'Error translating text. Please try again.');
    } finally {
      setTranslating(false);
    }
  };

  const playTTS = async (text, language, label) => {
    setPlayingAudio(label);
    try {
      const response = await axios.post('http://127.0.0.1:8000/tts', {
        text: text,
        language: language
      }, {
        responseType: 'blob'
      });
      
      const audioBlob = new Blob([response.data], { type: 'audio/mpeg' });
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      
      audio.onended = () => {
        setPlayingAudio(null);
        URL.revokeObjectURL(audioUrl);
      };
      
      audio.onerror = () => {
        setPlayingAudio(null);
        setError('Error playing audio');
        URL.revokeObjectURL(audioUrl);
      };
      
      await audio.play();
    } catch (err) {
      setPlayingAudio(null);
      setError('Error generating audio: ' + (err.response?.data?.detail || err.message));
    }
  };

  const handleTransliterate = async (text) => {
    if (!text.trim() || !useTransliteration) {
      setPreviewTamil('');
      return;
    }

    try {
      const res = await axios.post('http://127.0.0.1:8000/transliterate', {
        text: text
      });
      setPreviewTamil(res.data.tamil);
    } catch (err) {
      setPreviewTamil('');
    }
  };

  const handleChat = async () => {
    if (!chatQuestion.trim() || !inputText.trim()) {
      setError('Please enter both article text and a question');
      return;
    }

    setChatLoading(true);
    setError('');

    try {
      const res = await axios.post('http://127.0.0.1:8000/chat', {
        article_text: inputText,
        question: chatQuestion,
        language: 'ta',
        transliterate: useTransliteration
      });
      
      const newChat = {
        originalQuestion: res.data.original_question,
        processedQuestion: res.data.processed_question,
        answer: res.data.answer,
        transliterated: res.data.transliterated,
        timestamp: new Date().toLocaleTimeString()
      };
      
      setChatHistory(prev => [...prev, newChat]);
      setChatQuestion('');
      setPreviewTamil('');
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || 'Error getting answer. Please try again.');
    } finally {
      setChatLoading(false);
    }
  };



  const clearAll = () => {
    setInputText('');
    setSummary('');
    setError('');
    setStats(null);
    setSummaryLanguage('tamil');
    setSummaryEnglish('');
    setTranslationMethod('');
    setPlayingAudio(null);
    setChatQuestion('');
    setChatHistory([]);
  };

  return (
    <div style={{
      maxWidth: '800px',
      margin: '20px auto',
      padding: '20px',
      fontFamily: 'Arial, sans-serif',
      backgroundColor: '#f9f9f9',
      borderRadius: '10px'
    }}>
      <header style={{ marginBottom: '30px', textAlign: 'center' }}>
        <div style={{ 
          display: 'grid', 
          gridTemplateRows: 'auto auto', 
          gap: '5px',
          justifyItems: 'center'
        }}>
          <h1 style={{ 
            background: 'linear-gradient(135deg, #e74c3c, #c0392b)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            fontSize: '48px',
            fontWeight: 'bold',
            margin: 0,
            textShadow: '2px 2px 4px rgba(0,0,0,0.1)',
            fontFamily: 'Georgia, serif'
          }}>சாரம்</h1>
          <p style={{ 
            color: '#7f8c8d', 
            fontSize: '16px', 
            margin: 0,
            fontStyle: 'italic'
          }}>- The Essence</p>
        </div>
      </header>

      <div style={{ marginBottom: '20px' }}>
        <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold', color: '#34495e' }}>
          Tamil News Text:
        </label>
        <textarea
          rows={8}
          style={{
            width: '100%',
            padding: '15px',
            border: '2px solid #ddd',
            borderRadius: '8px',
            fontSize: '16px',
            fontFamily: 'inherit',
            resize: 'vertical',
            outline: 'none',
            transition: 'border-color 0.3s'
          }}
          placeholder='தமிழ் செய்தியை இங்கே ஒட்டவும்... (Paste Tamil news here...)'
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onFocus={(e) => e.target.style.borderColor = '#3498db'}
          onBlur={(e) => e.target.style.borderColor = '#ddd'}
        />
      </div>



      <div style={{ marginBottom: '20px' }}>
        <div style={{ marginBottom: '15px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={useMyMemory}
              onChange={(e) => setUseMyMemory(e.target.checked)}
              style={{ transform: 'scale(1.2)' }}
            />
            <span style={{ color: '#2c3e50', fontSize: '14px', fontWeight: 'bold' }}>
              🌐 Use MyMemory API for Translation (Better Quality)
            </span>
          </label>
          <div style={{ fontSize: '12px', color: '#7f8c8d', marginLeft: '28px', marginTop: '2px' }}>
            {useMyMemory ? '✅ High-quality online translation' : '📖 Basic dictionary translation'}
          </div>
        </div>
        
        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            style={{
              padding: '12px 24px',
              backgroundColor: loading ? '#95a5a6' : '#3498db',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: '16px',
              fontWeight: 'bold',
              transition: 'background-color 0.3s'
            }}
            onClick={handleSummarize}
            disabled={loading}
          >
            {loading ? '⏳ Summarizing...' : '🔍 Summarize'}
          </button>
        
        {summary && (
          <button
            style={{
              padding: '12px 24px',
              backgroundColor: translating ? '#95a5a6' : '#f39c12',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: translating ? 'not-allowed' : 'pointer',
              fontSize: '16px',
              fontWeight: 'bold',
              transition: 'background-color 0.3s'
            }}
            onClick={handleTranslate}
            disabled={translating}
          >
            {translating ? '⏳ Translating...' : 
             summaryLanguage === 'tamil' ? '🌐 Translate to English' : '🌐 Show Tamil'}
          </button>
        )}

          
          <button
            style={{
              padding: '12px 24px',
              backgroundColor: '#e74c3c',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '16px',
              fontWeight: 'bold'
            }}
            onClick={clearAll}
          >
            🗑️ Clear
          </button>
        </div>
      </div>

      {error && (
        <div style={{
          padding: '15px',
          backgroundColor: '#ffebee',
          border: '1px solid #f44336',
          borderRadius: '6px',
          color: '#c62828',
          marginBottom: '20px'
        }}>
          ❌ {error}
        </div>
      )}

      {summary && (
        <div style={{
          marginTop: '30px',
          padding: '20px',
          backgroundColor: 'white',
          border: '2px solid #27ae60',
          borderRadius: '8px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '15px' }}>
            <h3 style={{ color: '#27ae60', margin: 0 }}>📝 Summary ({summaryLanguage === 'tamil' ? 'Tamil' : 'English'})</h3>
            <button
              style={{
                padding: '6px 12px',
                backgroundColor: playingAudio === 'summary' ? '#95a5a6' : '#e67e22',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: playingAudio === 'summary' ? 'not-allowed' : 'pointer',
                fontSize: '12px',
                fontWeight: 'bold'
              }}
              onClick={() => playTTS(summaryLanguage === 'tamil' ? summary : summaryEnglish, summaryLanguage === 'tamil' ? 'ta' : 'en', 'summary')}
              disabled={playingAudio === 'summary'}
            >
              {playingAudio === 'summary' ? '🔊 Playing...' : `🔊 Play ${summaryLanguage === 'tamil' ? 'Tamil' : 'English'}`}
            </button>
          </div>
          <p style={{
            fontSize: '16px',
            lineHeight: '1.6',
            color: '#2c3e50',
            marginBottom: '15px'
          }}>
            {summaryLanguage === 'tamil' ? summary : summaryEnglish}
          </p>
          
          {stats && (
            <div style={{
              padding: '10px',
              backgroundColor: '#f8f9fa',
              borderRadius: '4px',
              fontSize: '14px',
              color: '#6c757d'
            }}>
              📊 Original: {stats.originalLength} chars | Summary: {stats.summaryLength} chars | 
              Compression: {stats.compressionRatio}% | Model: {stats.mode}
              {translationMethod && summaryLanguage === 'english' && (
                <span> | Translation: {translationMethod}</span>
              )}
            </div>
          )}
          

        </div>
      )}

      {inputText && (
        <div style={{
          marginTop: '30px',
          padding: '20px',
          backgroundColor: 'white',
          border: '2px solid #9b59b6',
          borderRadius: '8px'
        }}>
          <h3 style={{ color: '#9b59b6', marginBottom: '15px' }}>🤖 Ask Questions About the Article</h3>
          
          <div style={{ marginBottom: '10px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={useTransliteration}
                onChange={(e) => {
                  setUseTransliteration(e.target.checked);
                  if (!e.target.checked) setPreviewTamil('');
                  else handleTransliterate(chatQuestion);
                }}
                style={{ transform: 'scale(1.1)' }}
              />
              <span style={{ color: '#7f8c8d', fontSize: '14px', fontWeight: 'bold' }}>
                🌐 Azhagi Mode (Type in English, get Tamil)
              </span>
            </label>
          </div>
          
          <div style={{ display: 'flex', gap: '10px', marginBottom: '15px' }}>
            <input
              type="text"
              style={{
                flex: 1,
                padding: '12px',
                border: '2px solid #ddd',
                borderRadius: '6px',
                fontSize: '14px',
                outline: 'none'
              }}
              placeholder={useTransliteration ? "yaar pesinar? enna nadanthathu? (Type in English...)" : "கேள்விகளை கேட்கவும்... (Ask questions in Tamil...)"}
              value={chatQuestion}
              onChange={(e) => {
                setChatQuestion(e.target.value);
                if (useTransliteration) {
                  handleTransliterate(e.target.value);
                }
              }}
              onKeyPress={(e) => e.key === 'Enter' && !chatLoading && handleChat()}
            />
            <button
              style={{
                padding: '12px 20px',
                backgroundColor: chatLoading ? '#95a5a6' : '#9b59b6',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: chatLoading ? 'not-allowed' : 'pointer',
                fontSize: '14px',
                fontWeight: 'bold'
              }}
              onClick={handleChat}
              disabled={chatLoading}
            >
              {chatLoading ? '⏳ Thinking...' : '💬 Ask'}
            </button>
          </div>
          
          {useTransliteration && previewTamil && (
            <div style={{
              padding: '8px 12px',
              backgroundColor: '#f0f8ff',
              border: '1px solid #b3d9ff',
              borderRadius: '4px',
              marginBottom: '10px',
              fontSize: '14px'
            }}>
              <span style={{ color: '#666', fontSize: '12px' }}>Tamil Preview: </span>
              <span style={{ color: '#2c3e50', fontWeight: 'bold' }}>{previewTamil}</span>
            </div>
          )}

          {chatHistory.length > 0 && (
            <div style={{
              maxHeight: '300px',
              overflowY: 'auto',
              border: '1px solid #eee',
              borderRadius: '6px',
              padding: '10px'
            }}>
              {chatHistory.map((chat, index) => (
                <div key={index} style={{ marginBottom: '15px', paddingBottom: '10px', borderBottom: '1px solid #f0f0f0' }}>
                  <div style={{
                    padding: '8px 12px',
                    backgroundColor: '#f8f9fa',
                    borderRadius: '4px',
                    marginBottom: '5px',
                    fontSize: '14px',
                    fontWeight: 'bold',
                    color: '#495057'
                  }}>
                    💬 Q: {chat.originalQuestion}
                  </div>
                  <div style={{
                    padding: '8px 12px',
                    backgroundColor: '#e8f5e8',
                    borderRadius: '4px',
                    fontSize: '14px',
                    lineHeight: '1.4',
                    color: '#2d5a2d'
                  }}>
                    🤖 A: {chat.answer}
                  </div>
                  <div style={{
                    fontSize: '11px',
                    color: '#999',
                    textAlign: 'right',
                    marginTop: '5px'
                  }}>
                    {chat.timestamp}
                  </div>
                </div>
              ))}
            </div>
          )}

          <div style={{
            fontSize: '12px',
            color: '#7f8c8d',
            marginTop: '10px',
            fontStyle: 'italic'
          }}>
            💡 Tip: Ask questions about people, places, events, or details mentioned in the article
          </div>
        </div>
      )}




    </div>
  );
}