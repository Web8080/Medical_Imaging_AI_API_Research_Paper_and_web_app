import React, { useState } from 'react';
import { 
  Card, 
  Upload, 
  Button, 
  Select, 
  Slider, 
  Row, 
  Col, 
  message, 
  Progress,
  Alert,
  Divider,
  Tag,
  Statistic
} from 'antd';
import { UploadOutlined, EyeOutlined, DownloadOutlined } from '@ant-design/icons';
import axios from 'axios';
import Plot from 'react-plotly.js';

const { Dragger } = Upload;
const { Option } = Select;

const ImageAnalysis = ({ apiToken, onAnalysisComplete, loading, setLoading }) => {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [modelType, setModelType] = useState('chest');
  const [analysisType, setAnalysisType] = useState('classification');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [availableModels, setAvailableModels] = useState(['chest', 'derma', 'oct']);

  const modelInfo = {
    chest: { name: 'Chest X-ray Analysis', description: 'Analyzes chest X-rays for pathology detection' },
    derma: { name: 'Dermatology Analysis', description: 'Classifies skin lesions from dermatoscopic images' },
    oct: { name: 'Retinal OCT Analysis', description: 'Analyzes retinal OCT images for disease detection' }
  };

  const handleFileUpload = (info) => {
    const { file } = info;
    
    if (file.status === 'done' || file.status === 'uploading') {
      setUploadedFile(file);
      
      // Create image preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file.originFileObj);
    }
  };

  const handleAnalyze = async () => {
    if (!uploadedFile) {
      message.error('Please upload an image first');
      return;
    }

    setLoading(true);
    setAnalysisResult(null);

    try {
      const formData = new FormData();
      formData.append('file', uploadedFile.originFileObj);
      formData.append('model_type', modelType);
      formData.append('analysis_type', analysisType);
      formData.append('confidence_threshold', confidenceThreshold);

      const response = await axios.post('/analyze', formData, {
        headers: {
          'Authorization': `Bearer ${apiToken}`,
          'Content-Type': 'multipart/form-data'
        }
      });

      if (response.status === 200) {
        const result = response.data;
        setAnalysisResult(result);
        onAnalysisComplete(result);
        message.success('Analysis completed successfully!');
      } else {
        message.error('Analysis failed');
      }
    } catch (error) {
      console.error('Analysis error:', error);
      message.error(`Analysis failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

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

  return (
    <div>
      <Row gutter={[24, 24]}>
        {/* Upload and Configuration */}
        <Col xs={24} lg={12}>
          <Card title="Upload Medical Image" style={{ height: '100%' }}>
            <Dragger
              name="file"
              multiple={false}
              accept=".png,.jpg,.jpeg,.dcm,.nii,.nii.gz"
              customRequest={({ file, onSuccess }) => {
                onSuccess("ok");
              }}
              onChange={handleFileUpload}
              showUploadList={false}
            >
              <p className="ant-upload-drag-icon">
                <UploadOutlined />
              </p>
              <p className="ant-upload-text">Click or drag file to this area to upload</p>
              <p className="ant-upload-hint">
                Support for PNG, JPG, DICOM, and NIfTI formats
              </p>
            </Dragger>

            {imagePreview && (
              <div style={{ marginTop: 16, textAlign: 'center' }}>
                <img 
                  src={imagePreview} 
                  alt="Preview" 
                  style={{ maxWidth: '100%', maxHeight: 200, borderRadius: 6 }}
                />
                <p style={{ marginTop: 8, color: '#666' }}>
                  {uploadedFile?.name}
                </p>
              </div>
            )}

            <Divider />

            <div style={{ marginBottom: 16 }}>
              <label>Model Type:</label>
              <Select
                value={modelType}
                onChange={setModelType}
                style={{ width: '100%', marginTop: 8 }}
              >
                {availableModels.map(model => (
                  <Option key={model} value={model}>
                    {modelInfo[model]?.name || model}
                  </Option>
                ))}
              </Select>
              <p style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
                {modelInfo[modelType]?.description}
              </p>
            </div>

            <div style={{ marginBottom: 16 }}>
              <label>Analysis Type:</label>
              <Select
                value={analysisType}
                onChange={setAnalysisType}
                style={{ width: '100%', marginTop: 8 }}
              >
                <Option value="classification">Classification</Option>
                <Option value="segmentation">Segmentation</Option>
              </Select>
            </div>

            <div style={{ marginBottom: 16 }}>
              <label>Confidence Threshold: {confidenceThreshold}</label>
              <Slider
                min={0}
                max={1}
                step={0.1}
                value={confidenceThreshold}
                onChange={setConfidenceThreshold}
                style={{ marginTop: 8 }}
              />
            </div>

            <Button
              type="primary"
              size="large"
              onClick={handleAnalyze}
              disabled={!uploadedFile || loading}
              loading={loading}
              style={{ width: '100%' }}
              className="analyze-button"
            >
              {loading ? 'Analyzing...' : 'Analyze Image'}
            </Button>
          </Card>
        </Col>

        {/* Results */}
        <Col xs={24} lg={12}>
          <Card title="Analysis Results" style={{ height: '100%' }}>
            {analysisResult ? (
              <div>
                {/* Basic Info */}
                <Row gutter={16} style={{ marginBottom: 16 }}>
                  <Col span={12}>
                    <Statistic
                      title="Processing Time"
                      value={analysisResult.processing_time}
                      suffix="s"
                      precision={2}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="Confidence"
                      value={analysisResult.confidence}
                      precision={3}
                    />
                  </Col>
                </Row>

                <Tag color={getConfidenceColor(analysisResult.confidence)}>
                  {getConfidenceLabel(analysisResult.confidence)} Confidence
                </Tag>

                <Divider />

                {/* Predictions */}
                {analysisResult.results?.top_prediction && (
                  <div style={{ marginBottom: 16 }}>
                    <h4>Top Prediction</h4>
                    <Alert
                      message={analysisResult.results.top_prediction.class}
                      description={`Confidence: ${(analysisResult.results.top_prediction.confidence * 100).toFixed(1)}%`}
                      type="success"
                      showIcon
                    />
                  </div>
                )}

                {/* All Predictions Chart */}
                {analysisResult.results?.predictions && (
                  <div>
                    <h4>All Predictions</h4>
                    <PredictionsChart predictions={analysisResult.results.predictions} />
                  </div>
                )}

                {/* Segmentation Results */}
                {analysisResult.results?.measurements && (
                  <div style={{ marginTop: 16 }}>
                    <h4>Segmentation Measurements</h4>
                    <Row gutter={16}>
                      <Col span={8}>
                        <Statistic
                          title="Area"
                          value={analysisResult.results.measurements.area}
                          suffix="px"
                        />
                      </Col>
                      <Col span={8}>
                        <Statistic
                          title="Density"
                          value={analysisResult.results.measurements.density}
                          precision={3}
                        />
                      </Col>
                      <Col span={8}>
                        <Statistic
                          title="BBox Area"
                          value={analysisResult.results.measurements.bounding_box_area}
                          suffix="px"
                        />
                      </Col>
                    </Row>
                  </div>
                )}

                {/* Request Info */}
                <Divider />
                <div style={{ fontSize: 12, color: '#666' }}>
                  <p><strong>Request ID:</strong> {analysisResult.request_id}</p>
                  <p><strong>Model:</strong> {analysisResult.model_used}</p>
                  <p><strong>Timestamp:</strong> {new Date(analysisResult.timestamp).toLocaleString()}</p>
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '40px 0', color: '#666' }}>
                <EyeOutlined style={{ fontSize: 48, marginBottom: 16 }} />
                <p>Upload an image and click "Analyze Image" to see results here</p>
              </div>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
};

const PredictionsChart = ({ predictions }) => {
  const data = predictions
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 10); // Show top 10 predictions

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

export default ImageAnalysis;
