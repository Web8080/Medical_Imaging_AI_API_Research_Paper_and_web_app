import React, { useState } from 'react';
import { Card, List, Tag, Button, Modal, Row, Col, Statistic, Timeline, Empty } from 'antd';
import { EyeOutlined, DownloadOutlined, DeleteOutlined, ClockCircleOutlined } from '@ant-design/icons';
import Plot from 'react-plotly.js';

const ResultsHistory = ({ results }) => {
  const [selectedResult, setSelectedResult] = useState(null);
  const [modalVisible, setModalVisible] = useState(false);

  const getConfidenceColor = (confidence) => {
    if (confidence > 0.8) return 'success';
    if (confidence > 0.5) return 'warning';
    return 'error';
  };

  const getConfidenceLabel = (confidence) => {
    if (confidence > 0.8) return 'High';
    if (confidence > 0.5) return 'Medium';
    return 'Low';
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const handleViewDetails = (result) => {
    setSelectedResult(result);
    setModalVisible(true);
  };

  const handleExportResult = (result) => {
    const dataStr = JSON.stringify(result, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `analysis_result_${result.request_id}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  if (!results || results.length === 0) {
    return (
      <Card>
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description="No analysis results yet"
        >
          <p>Upload and analyze images to see results here</p>
        </Empty>
      </Card>
    );
  }

  return (
    <div>
      <Card title={`Analysis History (${results.length} results)`}>
        <List
          itemLayout="vertical"
          dataSource={results}
          renderItem={(result, index) => (
            <List.Item
              key={result.request_id}
              actions={[
                <Button 
                  type="primary" 
                  icon={<EyeOutlined />} 
                  onClick={() => handleViewDetails(result)}
                >
                  View Details
                </Button>,
                <Button 
                  icon={<DownloadOutlined />} 
                  onClick={() => handleExportResult(result)}
                >
                  Export
                </Button>
              ]}
            >
              <List.Item.Meta
                title={
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>Analysis #{results.length - index}</span>
                    <Tag color={getConfidenceColor(result.confidence)}>
                      {getConfidenceLabel(result.confidence)} Confidence
                    </Tag>
                  </div>
                }
                description={
                  <div>
                    <p><strong>Model:</strong> {result.model_used}</p>
                    <p><strong>Processing Time:</strong> {result.processing_time.toFixed(2)}s</p>
                    <p><strong>Timestamp:</strong> {formatTimestamp(result.timestamp)}</p>
                    {result.results?.top_prediction && (
                      <p><strong>Top Prediction:</strong> {result.results.top_prediction.class} 
                        ({result.results.top_prediction.confidence.toFixed(3)})
                      </p>
                    )}
                  </div>
                }
              />
            </List.Item>
          )}
        />
      </Card>

      <Modal
        title="Analysis Details"
        visible={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={null}
        width={800}
      >
        {selectedResult && <AnalysisDetailsModal result={selectedResult} />}
      </Modal>
    </div>
  );
};

const AnalysisDetailsModal = ({ result }) => {
  const getConfidenceColor = (confidence) => {
    if (confidence > 0.8) return '#52c41a';
    if (confidence > 0.5) return '#faad14';
    return '#ff4d4f';
  };

  return (
    <div>
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Statistic
            title="Processing Time"
            value={result.processing_time}
            suffix="s"
            precision={2}
          />
        </Col>
        <Col span={8}>
          <Statistic
            title="Confidence"
            value={result.confidence}
            precision={3}
            valueStyle={{ color: getConfidenceColor(result.confidence) }}
          />
        </Col>
        <Col span={8}>
          <Statistic
            title="Model"
            value={result.model_used}
          />
        </Col>
      </Row>

      <Timeline>
        <Timeline.Item dot={<ClockCircleOutlined />}>
          <strong>Request ID:</strong> {result.request_id}
        </Timeline.Item>
        <Timeline.Item>
          <strong>Timestamp:</strong> {new Date(result.timestamp).toLocaleString()}
        </Timeline.Item>
        <Timeline.Item>
          <strong>Status:</strong> {result.status}
        </Timeline.Item>
      </Timeline>

      {result.results?.top_prediction && (
        <div style={{ marginTop: 24 }}>
          <h4>Top Prediction</h4>
          <Card size="small">
            <p><strong>Class:</strong> {result.results.top_prediction.class}</p>
            <p><strong>Confidence:</strong> {(result.results.top_prediction.confidence * 100).toFixed(1)}%</p>
          </Card>
        </div>
      )}

      {result.results?.predictions && (
        <div style={{ marginTop: 24 }}>
          <h4>All Predictions</h4>
          <PredictionsChart predictions={result.results.predictions} />
        </div>
      )}

      {result.results?.measurements && (
        <div style={{ marginTop: 24 }}>
          <h4>Segmentation Measurements</h4>
          <Row gutter={16}>
            <Col span={8}>
              <Statistic
                title="Area"
                value={result.results.measurements.area}
                suffix="px"
              />
            </Col>
            <Col span={8}>
              <Statistic
                title="Density"
                value={result.results.measurements.density}
                precision={3}
              />
            </Col>
            <Col span={8}>
              <Statistic
                title="BBox Area"
                value={result.results.measurements.bounding_box_area}
                suffix="px"
              />
            </Col>
          </Row>
        </div>
      )}
    </div>
  );
};

const PredictionsChart = ({ predictions }) => {
  const data = predictions
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 10);

  const plotData = [{
    x: data.map(p => p.class),
    y: data.map(p => p.confidence),
    type: 'bar',
    marker: {
      color: data.map(p => {
        if (p.confidence > 0.8) return '#52c41a';
        if (p.confidence > 0.5) return '#faad14';
        return '#ff4d4f';
      })
    }
  }];

  const layout = {
    title: 'Prediction Confidence Scores',
    xaxis: { 
      title: 'Class',
      tickangle: -45
    },
    yaxis: { 
      title: 'Confidence',
      range: [0, 1]
    },
    margin: { t: 50, r: 30, b: 100, l: 50 },
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

export default ResultsHistory;
