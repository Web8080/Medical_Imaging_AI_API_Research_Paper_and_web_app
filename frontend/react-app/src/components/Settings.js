import React, { useState } from 'react';
import { Card, Form, Input, Button, Switch, Divider, Alert, message, Row, Col } from 'antd';
import { SettingOutlined, SaveOutlined, ReloadOutlined } from '@ant-design/icons';

const Settings = ({ apiToken, setApiToken }) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);

  const handleSave = async (values) => {
    setLoading(true);
    
    try {
      // Update API token
      setApiToken(values.apiToken);
      
      // Here you would typically save settings to localStorage or backend
      localStorage.setItem('apiToken', values.apiToken);
      localStorage.setItem('apiUrl', values.apiUrl);
      localStorage.setItem('autoRefresh', values.autoRefresh);
      localStorage.setItem('notifications', values.notifications);
      
      message.success('Settings saved successfully');
    } catch (error) {
      message.error('Failed to save settings');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    form.resetFields();
    message.info('Settings reset to defaults');
  };

  const handleTestConnection = async () => {
    const values = form.getFieldsValue();
    
    try {
      const response = await fetch(`${values.apiUrl}/health`);
      if (response.ok) {
        message.success('Connection test successful');
      } else {
        message.error('Connection test failed');
      }
    } catch (error) {
      message.error('Connection test failed: ' + error.message);
    }
  };

  return (
    <div>
      <Card title="Settings" icon={<SettingOutlined />}>
        <Form
          form={form}
          layout="vertical"
          initialValues={{
            apiToken: apiToken,
            apiUrl: 'http://localhost:8000',
            autoRefresh: true,
            notifications: true,
            refreshInterval: 30,
            maxHistoryItems: 100
          }}
          onFinish={handleSave}
        >
          <Divider orientation="left">API Configuration</Divider>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="API URL"
                name="apiUrl"
                rules={[{ required: true, message: 'Please enter API URL' }]}
              >
                <Input placeholder="http://localhost:8000" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="API Token"
                name="apiToken"
                rules={[{ required: true, message: 'Please enter API token' }]}
              >
                <Input.Password placeholder="Enter your API token" />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item>
            <Button 
              type="primary" 
              onClick={handleTestConnection}
              style={{ marginRight: 8 }}
            >
              Test Connection
            </Button>
          </Form.Item>

          <Divider orientation="left">Dashboard Settings</Divider>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="Auto Refresh"
                name="autoRefresh"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="Notifications"
                name="notifications"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="Refresh Interval (seconds)"
                name="refreshInterval"
                rules={[{ required: true, message: 'Please enter refresh interval' }]}
              >
                <Input type="number" min={10} max={300} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="Max History Items"
                name="maxHistoryItems"
                rules={[{ required: true, message: 'Please enter max history items' }]}
              >
                <Input type="number" min={10} max={1000} />
              </Form.Item>
            </Col>
          </Row>

          <Divider orientation="left">Advanced Settings</Divider>

          <Form.Item
            label="Request Timeout (seconds)"
            name="requestTimeout"
            initialValue={30}
          >
            <Input type="number" min={5} max={120} />
          </Form.Item>

          <Form.Item
            label="Max File Size (MB)"
            name="maxFileSize"
            initialValue={50}
          >
            <Input type="number" min={1} max={500} />
          </Form.Item>

          <Form.Item
            label="Supported Formats"
            name="supportedFormats"
            initialValue="PNG, JPG, JPEG, DICOM, NIfTI"
          >
            <Input disabled />
          </Form.Item>

          <Divider />

          <Form.Item>
            <Button 
              type="primary" 
              htmlType="submit" 
              icon={<SaveOutlined />}
              loading={loading}
              style={{ marginRight: 8 }}
            >
              Save Settings
            </Button>
            <Button 
              onClick={handleReset}
              icon={<ReloadOutlined />}
            >
              Reset to Defaults
            </Button>
          </Form.Item>
        </Form>
      </Card>

      <Card title="System Information" style={{ marginTop: 24 }}>
        <Row gutter={[16, 16]}>
          <Col span={8}>
            <div>
              <strong>Dashboard Version:</strong> 1.0.0
            </div>
          </Col>
          <Col span={8}>
            <div>
              <strong>React Version:</strong> 18.2.0
            </div>
          </Col>
          <Col span={8}>
            <div>
              <strong>Ant Design Version:</strong> 4.24.0
            </div>
          </Col>
        </Row>
        
        <Divider />
        
        <Alert
          message="About This Dashboard"
          description="This dashboard provides a user-friendly interface for the Medical Imaging AI API. It supports image upload, analysis, and result visualization for medical imaging applications."
          type="info"
          showIcon
        />
      </Card>
    </div>
  );
};

export default Settings;
