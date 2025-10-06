import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Button, Spin, Alert, Progress } from 'antd';
import { ReloadOutlined, CheckCircleOutlined, CloseCircleOutlined, ClockCircleOutlined } from '@ant-design/icons';
import axios from 'axios';
import Plot from 'react-plotly.js';

const SystemMetrics = ({ apiToken }) => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchMetrics = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.get('/metrics', {
        headers: {
          'Authorization': `Bearer ${apiToken}`
        }
      });
      
      if (response.status === 200) {
        setMetrics(response.data);
      } else {
        setError('Failed to fetch metrics');
      }
    } catch (err) {
      setError(`Error: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  const getSuccessRate = () => {
    if (!metrics || metrics.total_requests === 0) return 0;
    return (metrics.successful_requests / metrics.total_requests) * 100;
  };

  const getFailureRate = () => {
    if (!metrics || metrics.total_requests === 0) return 0;
    return (metrics.failed_requests / metrics.total_requests) * 100;
  };

  if (error) {
    return (
      <Card>
        <Alert
          message="Error Loading Metrics"
          description={error}
          type="error"
          showIcon
          action={
            <Button size="small" onClick={fetchMetrics}>
              Retry
            </Button>
          }
        />
      </Card>
    );
  }

  return (
    <div>
      <Card 
        title="System Performance Metrics" 
        extra={
          <Button 
            icon={<ReloadOutlined />} 
            onClick={fetchMetrics}
            loading={loading}
          >
            Refresh
          </Button>
        }
      >
        <Spin spinning={loading}>
          {metrics ? (
            <div>
              {/* Overview Metrics */}
              <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
                <Col xs={24} sm={12} lg={6}>
                  <Card size="small">
                    <Statistic
                      title="Total Requests"
                      value={metrics.total_requests}
                      prefix={<CheckCircleOutlined />}
                    />
                  </Card>
                </Col>
                <Col xs={24} sm={12} lg={6}>
                  <Card size="small">
                    <Statistic
                      title="Successful Requests"
                      value={metrics.successful_requests}
                      valueStyle={{ color: '#52c41a' }}
                      prefix={<CheckCircleOutlined />}
                    />
                  </Card>
                </Col>
                <Col xs={24} sm={12} lg={6}>
                  <Card size="small">
                    <Statistic
                      title="Failed Requests"
                      value={metrics.failed_requests}
                      valueStyle={{ color: '#ff4d4f' }}
                      prefix={<CloseCircleOutlined />}
                    />
                  </Card>
                </Col>
                <Col xs={24} sm={12} lg={6}>
                  <Card size="small">
                    <Statistic
                      title="Avg Processing Time"
                      value={metrics.average_processing_time}
                      suffix="s"
                      precision={2}
                      prefix={<ClockCircleOutlined />}
                    />
                  </Card>
                </Col>
              </Row>

              {/* Success Rate Progress */}
              <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
                <Col span={24}>
                  <Card size="small" title="Success Rate">
                    <Progress
                      percent={getSuccessRate()}
                      strokeColor={{
                        '0%': '#ff4d4f',
                        '50%': '#faad14',
                        '100%': '#52c41a',
                      }}
                      format={(percent) => `${percent.toFixed(1)}%`}
                    />
                    <div style={{ marginTop: 8, fontSize: 12, color: '#666' }}>
                      {metrics.successful_requests} successful out of {metrics.total_requests} total requests
                    </div>
                  </Card>
                </Col>
              </Row>

              {/* Model Performance */}
              {metrics.model_performance && Object.keys(metrics.model_performance).length > 0 && (
                <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
                  <Col span={24}>
                    <Card size="small" title="Model Performance">
                      <ModelPerformanceChart modelPerformance={metrics.model_performance} />
                    </Card>
                  </Col>
                </Row>
              )}

              {/* Additional Metrics */}
              <Row gutter={[16, 16]}>
                <Col xs={24} sm={12}>
                  <Card size="small" title="System Information">
                    <div style={{ fontSize: 14 }}>
                      <p><strong>Requests per Minute:</strong> {metrics.requests_per_minute || 0}</p>
                      <p><strong>Uptime:</strong> {formatUptime(metrics.uptime_seconds)}</p>
                      <p><strong>Last Updated:</strong> {new Date(metrics.timestamp).toLocaleString()}</p>
                    </div>
                  </Card>
                </Col>
                <Col xs={24} sm={12}>
                  <Card size="small" title="Error Summary">
                    {metrics.error_counts && Object.keys(metrics.error_counts).length > 0 ? (
                      <div>
                        {Object.entries(metrics.error_counts).map(([model, count]) => (
                          <div key={model} style={{ marginBottom: 8 }}>
                            <span style={{ fontWeight: 'bold' }}>{model}:</span> {count} errors
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div style={{ color: '#52c41a' }}>No errors recorded</div>
                    )}
                  </Card>
                </Col>
              </Row>
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '40px 0' }}>
              <p>No metrics data available</p>
              <Button onClick={fetchMetrics}>Load Metrics</Button>
            </div>
          )}
        </Spin>
      </Card>
    </div>
  );
};

const ModelPerformanceChart = ({ modelPerformance }) => {
  const models = Object.keys(modelPerformance);
  const successRates = models.map(model => modelPerformance[model].success_rate * 100);
  const avgTimes = models.map(model => modelPerformance[model].average_processing_time);

  const plotData = [
    {
      x: models,
      y: successRates,
      type: 'bar',
      name: 'Success Rate (%)',
      marker: { color: '#52c41a' }
    }
  ];

  const layout = {
    title: 'Model Success Rates',
    xaxis: { title: 'Model' },
    yaxis: { title: 'Success Rate (%)', range: [0, 100] },
    height: 300
  };

  return (
    <Plot
      data={plotData}
      layout={layout}
      style={{ width: '100%', height: '300px' }}
    />
  );
};

const formatUptime = (seconds) => {
  if (!seconds) return 'Unknown';
  
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  
  if (days > 0) {
    return `${days}d ${hours}h ${minutes}m`;
  } else if (hours > 0) {
    return `${hours}h ${minutes}m`;
  } else {
    return `${minutes}m`;
  }
};

export default SystemMetrics;
