// This file will be served from Flask's static directory
// It contains the full React component for the dashboard

// Icon components
const RefreshIcon = ({ spinning }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" 
         stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" 
         className={`mr-2 ${spinning ? 'animate-spin' : ''}`}>
      <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"></path>
      <path d="M21 3v5h-5"></path>
      <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"></path>
      <path d="M3 21v-5h5"></path>
    </svg>
  );
  
  const ArrowUpIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" 
         stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" 
         className="text-green-600">
      <path d="m12 19-7-7 7-7 7 7-7 7z"></path>
    </svg>
  );
  
  const ArrowDownIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" 
         stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" 
         className="text-red-600">
      <path d="m19 12-7 7-7-7 7-7 7 7z"></path>
    </svg>
  );
  
  const MinusIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" 
         stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" 
         className="text-gray-600">
      <path d="M5 12h14"></path>
    </svg>
  );
  
  const ClockIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" 
         stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" 
         className="mr-2">
      <circle cx="12" cy="12" r="10"></circle>
      <polyline points="12 6 12 12 16 14"></polyline>
    </svg>
  );
  
  const ExternalLinkIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" 
         stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" 
         className="ml-1">
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
      <polyline points="15 3 21 3 21 9"></polyline>
      <line x1="10" y1="14" x2="21" y2="3"></line>
    </svg>
  );
  
  // Main Dashboard Component
  const StockNewsDashboard = () => {
    const [news, setNews] = React.useState([]);
    const [loading, setLoading] = React.useState(true);
    const [refreshing, setRefreshing] = React.useState(false);
    const [error, setError] = React.useState(null);
    const [lastUpdated, setLastUpdated] = React.useState('');
    const [visibleCount, setVisibleCount] = React.useState(10);
    const [searchTerm, setSearchTerm] = React.useState('');
  
    // Function to fetch news data
    const fetchNews = React.useCallback(async () => {
      try {
        setLoading(true);
        setError(null);
        
        const response = await fetch('/api/news');
        if (!response.ok) {
          throw new Error(`Error: ${response.status}`);
        }
        
        const data = await response.json();
        setNews(data.news);
        setLastUpdated(data.last_updated);
      } catch (err) {
        setError(`Failed to fetch news: ${err.message}`);
        console.error('Error fetching news:', err);
      } finally {
        setLoading(false);
      }
    }, []);
  
    // Function to manually refresh data
    const refreshData = React.useCallback(async () => {
      try {
        setRefreshing(true);
        setError(null);
        
        const response = await fetch('/api/update', {
          method: 'POST',
        });
        
        if (!response.ok) {
          throw new Error(`Error: ${response.status}`);
        }
        
        // After successful update, fetch the new data
        await fetchNews();
      } catch (err) {
        setError(`Failed to refresh data: ${err.message}`);
        console.error('Error refreshing data:', err);
      } finally {
        setRefreshing(false);
      }
    }, [fetchNews]);
  
    // Initial data fetch
    React.useEffect(() => {
      fetchNews();
      
      // Auto-refresh every 5 minutes
      const interval = setInterval(fetchNews, 5 * 60 * 1000);
      return () => clearInterval(interval);
    }, [fetchNews]);
  
    // Function to determine color based on sentiment
    const getSentimentColor = (sentiment, score) => {
      // Make sure score is a number
      const numScore = parseFloat(score);
      const intensity = Math.min(Math.max(numScore, 0.3), 1); // Ensure intensity is between 0.3 and 1
      
      if (sentiment === 'positive') {
        // Green with varying opacity
        return `rgba(0, 128, 0, ${intensity})`;
      } else if (sentiment === 'negative') {
        // Red with varying opacity
        return `rgba(220, 20, 60, ${intensity})`;
      } else {
        // Grey with varying opacity
        return `rgba(128, 128, 128, ${intensity})`;
      }
    };
  
    // Function to get sentiment icon
    const SentimentIcon = ({ sentiment }) => {
      if (sentiment === 'positive') {
        return <ArrowUpIcon />;
      } else if (sentiment === 'negative') {
        return <ArrowDownIcon />;
      } else {
        return <MinusIcon />;
      }
    };
  
    // Load more news
    const loadMore = () => {
      setVisibleCount(prev => prev + 10);
    };
  
    // Filter news based on search term
    const filteredNews = React.useMemo(() => {
      if (!searchTerm) return news;
      
      return news.filter(item => 
        item.title.toLowerCase().includes(searchTerm.toLowerCase()) || 
        item.description.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }, [news, searchTerm]);
  
    return (
      <div className="min-h-screen bg-gray-100 p-4 md:p-8">
        <div className="max-w-6xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center mb-6">
            <h1 className="text-3xl font-bold text-gray-800 mb-4 md:mb-0">Stock News Dashboard</h1>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center text-gray-600">
                <ClockIcon />
                <span className="text-sm">
                  {lastUpdated ? `Last updated: ${lastUpdated}` : 'Never updated'}
                </span>
              </div>
              
              <button 
                onClick={refreshData}
                disabled={refreshing}
                className="flex items-center bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-all refresh-button"
              >
                <RefreshIcon spinning={refreshing} />
                {refreshing ? 'Updating...' : 'Get Latest News'}
              </button>
            </div>
          </div>
  
          <div className="mb-6">
            <input
              type="text"
              placeholder="Search news..."
              className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
  
          {error && (
            <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6 rounded" role="alert">
              <p>{error}</p>
            </div>
          )}
  
          {loading && !refreshing ? (
            <div className="grid grid-cols-1 gap-6 animate-pulse">
              {[...Array(5)].map((_, idx) => (
                <div key={idx} className="bg-white rounded-xl shadow-lg p-6 h-48"></div>
              ))}
            </div>
          ) : (
            <>
              <div className="grid grid-cols-1 gap-6">
                {filteredNews.slice(0, visibleCount).map((item, index) => {
                  const scoreValue = parseFloat(
                    item.sentiment === 'positive' ? item.positive_Score : 
                    item.sentiment === 'negative' ? item.negative_Score : 
                    item.neutral_Score
                  );
                  
                  return (
                    <div 
                      key={index}
                      className="bg-white rounded-xl shadow-lg overflow-hidden transition-all duration-300 hover:shadow-xl transform hover:-translate-y-1 news-card"
                      style={{
                        animationDelay: `${index * 0.05}s`,
                        borderLeft: `6px solid ${getSentimentColor(item.sentiment, scoreValue)}`
                      }}
                    >
                      <div className="p-6">
                        <div className="flex justify-between mb-2">
                          <div className="flex items-center">
                            <SentimentIcon sentiment={item.sentiment} />
                            <span 
                              className={`ml-2 text-sm font-medium px-2 py-1 rounded ${
                                item.sentiment === 'positive' ? 'bg-green-100 text-green-800' : 
                                item.sentiment === 'negative' ? 'bg-red-100 text-red-800' : 
                                'bg-gray-100 text-gray-800'
                              }`}
                            >
                              {item.sentiment.charAt(0).toUpperCase() + item.sentiment.slice(1)}
                              {' '}{(scoreValue * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="text-gray-500 text-sm">{item.date}</div>
                        </div>
                        
                        <h2 className="text-xl font-semibold mb-3 text-gray-800">{item.title}</h2>
                        
                        <p className="text-gray-600 mb-4">{item.description}</p>
                        
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-gray-500">{item.source}</span>
                          <a 
                            href={item.url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="flex items-center text-blue-600 hover:text-blue-800 text-sm font-medium"
                          >
                            Read more <ExternalLinkIcon />
                          </a>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
              
              {filteredNews.length > visibleCount && (
                <div className="mt-8 text-center">
                  <button 
                    onClick={loadMore}
                    className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-medium py-2 px-6 rounded-lg transition-colors"
                  >
                    Load More News
                  </button>
                </div>
              )}
  
              {filteredNews.length === 0 && (
                <div className="text-center py-12">
                  <p className="text-gray-500 text-lg">No news matching your search.</p>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    );
  };
  
  // Render the dashboard
  ReactDOM.render(<StockNewsDashboard />, document.getElementById('root'));