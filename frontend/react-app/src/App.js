import React, { useState, useEffect } from 'react';
import { Layout, Menu, Card, Row, Col, Button, Upload, message, Spin, Alert } from 'antd';
import { 
  UploadOutlined, 
  BarChartOutlined, 
  SettingOutlined, 
  FileImageOutlined,
  DashboardOutlined,
  ApiOutlined
} from '@ant-design/icons';
import axios from 'axios';
import './App.css';

// Import components
import ImageAnalysis from './components/ImageAnalysis';
import ResultsHistory from './components/ResultsHistory';
import SystemMetrics from './components/SystemMetrics';
import Settings from './components/Settings';
import DICOMViewer from './components/DICOMViewer';

const { Header, Sider, Content } = Layout;

function App() {
  const [selectedKey, setSelectedKey] = useState('analysis');
  const [apiStatus, setApiStatus] = useState('checking');
  const [apiToken, setApiToken] = useState('test_token');
  const [analysisResults, setAnalysisResults] = useState([]);
  const [loading, setLoading] = useState(false);

  // Check API health on component mount
  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await axios.get('/health');
      if (response.status === 200) {
        setApiStatus('online');
      } else {
        setApiStatus('offline');
      }
    } catch (error) {
      setApiStatus('offline');
    }
  };

  const handleMenuClick = ({ key }) => {
    setSelectedKey(key);
  };

  const handleAnalysisComplete = (result) => {
    setAnalysisResults(prev => [result, ...prev]);
  };

  const renderContent = () => {
    switch (selectedKey) {
      case 'analysis':
        return (
          <ImageAnalysis 
            apiToken={apiToken}
            onAnalysisComplete={handleAnalysisComplete}
            loading={loading}
            setLoading={setLoading}
          />
        );
      case 'history':
        return <ResultsHistory results={analysisResults} />;
      case 'metrics':
        return <SystemMetrics apiToken={apiToken} />;
      case 'dicom':
        return <DICOMViewer />;
      case 'settings':
        return <Settings apiToken={apiToken} setApiToken={setApiToken} />;
      default:
        return <ImageAnalysis 
          apiToken={apiToken}
          onAnalysisComplete={handleAnalysisComplete}
          loading={loading}
          setLoading={setLoading}
        />;
    }
  };

  const menuItems = [
    {
      key: 'analysis',
      icon: <FileImageOutlined />,
      label: 'Image Analysis',
    },
    {
      key: 'history',
      icon: <BarChartOutlined />,
      label: 'Results History',
    },
    {
      key: 'metrics',
      icon: <DashboardOutlined />,
      label: 'System Metrics',
    },
    {
      key: 'dicom',
      icon: <ApiOutlined />,
      label: 'DICOM Viewer',
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: 'Settings',
    },
  ];

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#001529', padding: '0 24px' }}>
        <div style={{ color: 'white', fontSize: '20px', fontWeight: 'bold' }}>
          Medical Imaging AI Dashboard
        </div>
        <div style={{ color: '#1890ff', fontSize: '14px' }}>
          Advanced AI-powered Medical Image Analysis
        </div>
      </Header>
      
      <Layout>
        <Sider width={250} style={{ background: '#fff' }}>
          <div style={{ padding: '16px', borderBottom: '1px solid #f0f0f0' }}>
            <div style={{ marginBottom: '8px', fontWeight: 'bold' }}>
              API Status
            </div>
            <div style={{ 
              color: apiStatus === 'online' ? '#52c41a' : '#ff4d4f',
              fontSize: '12px'
            }}>
              {apiStatus === 'online' ? 'Online' : 'Offline'}
            </div>
            {apiStatus === 'offline' && (
              <Alert
                message="API Offline"
                description="Please ensure the API server is running on localhost:8000"
                type="warning"
                showIcon
                style={{ marginTop: '8px', fontSize: '12px' }}
              />
            )}
          </div>
          
          <Menu
            mode="inline"
            selectedKeys={[selectedKey]}
            onClick={handleMenuClick}
            items={menuItems}
            style={{ borderRight: 0 }}
          />
        </Sider>
        
        <Layout style={{ padding: '24px' }}>
          <Content style={{ background: '#fff', padding: '24px', minHeight: 280 }}>
            <Spin spinning={loading} tip="Processing...">
              {renderContent()}
            </Spin>
          </Content>
        </Layout>
      </Layout>
    </Layout>
  );
}

export default App;
