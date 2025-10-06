import React, { useEffect, useRef, useState } from 'react';
import { Card, Button, Upload, message, Row, Col, Slider, Select } from 'antd';
import { UploadOutlined, ZoomInOutlined, ZoomOutOutlined, RotateLeftOutlined, RotateRightOutlined } from '@ant-design/icons';
import * as cornerstone from 'cornerstone-core';
import * as cornerstoneTools from 'cornerstone-tools';
import cornerstoneWADOImageLoader from 'cornerstone-wado-image-loader';

const { Dragger } = Upload;

const DICOMViewer = () => {
  const canvasRef = useRef(null);
  const [imageId, setImageId] = useState(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [windowWidth, setWindowWidth] = useState(400);
  const [windowCenter, setWindowCenter] = useState(40);
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState(0);

  useEffect(() => {
    // Initialize cornerstone
    const element = canvasRef.current;
    if (element) {
      cornerstone.enable(element);
      
      // Configure tools
      cornerstoneTools.init();
      
      // Add tools
      cornerstoneTools.addTool(cornerstoneTools.PanTool);
      cornerstoneTools.addTool(cornerstoneTools.ZoomTool);
      cornerstoneTools.addTool(cornerstoneTools.WwwcTool);
      cornerstoneTools.addTool(cornerstoneTools.RotateTool);
      
      // Set active tools
      cornerstoneTools.setToolActive('Pan', { mouseButtonMask: 4 });
      cornerstoneTools.setToolActive('Zoom', { mouseButtonMask: 2 });
      cornerstoneTools.setToolActive('Wwwc', { mouseButtonMask: 1 });
      cornerstoneTools.setToolActive('Rotate', { mouseButtonMask: 1 });
    }

    return () => {
      if (element) {
        cornerstone.disable(element);
      }
    };
  }, []);

  const handleFileUpload = (info) => {
    const { file } = info;
    
    if (file.status === 'done' || file.status === 'uploading') {
      const fileObj = file.originFileObj;
      
      // Create a URL for the file
      const url = URL.createObjectURL(fileObj);
      
      // Configure the image loader
      cornerstoneWADOImageLoader.external.cornerstone = cornerstone;
      cornerstoneWADOImageLoader.external.dicomParser = window.dicomParser;
      
      // Load the image
      const imageId = `wadouri:${url}`;
      setImageId(imageId);
      
      cornerstone.loadImage(imageId).then((image) => {
        const element = canvasRef.current;
        cornerstone.displayImage(element, image);
        setIsLoaded(true);
        
        // Set initial window/level
        const viewport = cornerstone.getViewport(element);
        viewport.voi.windowWidth = windowWidth;
        viewport.voi.windowCenter = windowCenter;
        cornerstone.setViewport(element, viewport);
        
        message.success('DICOM image loaded successfully');
      }).catch((error) => {
        console.error('Error loading DICOM:', error);
        message.error('Failed to load DICOM image');
      });
    }
  };

  const handleWindowWidthChange = (value) => {
    setWindowWidth(value);
    if (isLoaded && imageId) {
      const element = canvasRef.current;
      const viewport = cornerstone.getViewport(element);
      viewport.voi.windowWidth = value;
      cornerstone.setViewport(element, viewport);
    }
  };

  const handleWindowCenterChange = (value) => {
    setWindowCenter(value);
    if (isLoaded && imageId) {
      const element = canvasRef.current;
      const viewport = cornerstone.getViewport(element);
      viewport.voi.windowCenter = value;
      cornerstone.setViewport(element, viewport);
    }
  };

  const handleZoomChange = (value) => {
    setZoom(value);
    if (isLoaded && imageId) {
      const element = canvasRef.current;
      const viewport = cornerstone.getViewport(element);
      viewport.scale = value;
      cornerstone.setViewport(element, viewport);
    }
  };

  const handleRotate = (direction) => {
    const newRotation = direction === 'left' ? rotation - 90 : rotation + 90;
    setRotation(newRotation);
    
    if (isLoaded && imageId) {
      const element = canvasRef.current;
      const viewport = cornerstone.getViewport(element);
      viewport.rotation = newRotation;
      cornerstone.setViewport(element, viewport);
    }
  };

  const resetView = () => {
    if (isLoaded && imageId) {
      const element = canvasRef.current;
      const viewport = cornerstone.getViewport(element);
      viewport.scale = 1;
      viewport.translation.x = 0;
      viewport.translation.y = 0;
      viewport.rotation = 0;
      cornerstone.setViewport(element, viewport);
      
      setZoom(1);
      setRotation(0);
    }
  };

  return (
    <div>
      <Row gutter={[24, 24]}>
        <Col xs={24} lg={18}>
          <Card title="DICOM Viewer" style={{ height: '100%' }}>
            <div style={{ marginBottom: 16 }}>
              <Dragger
                name="file"
                multiple={false}
                accept=".dcm,.dicom"
                customRequest={({ file, onSuccess }) => {
                  onSuccess("ok");
                }}
                onChange={handleFileUpload}
                showUploadList={false}
              >
                <p className="ant-upload-drag-icon">
                  <UploadOutlined />
                </p>
                <p className="ant-upload-text">Click or drag DICOM file to this area to upload</p>
                <p className="ant-upload-hint">
                  Support for DICOM (.dcm, .dicom) files
                </p>
              </Dragger>
            </div>

            <div 
              ref={canvasRef}
              className="dicom-viewer"
              style={{
                width: '100%',
                height: '500px',
                border: '1px solid #d9d9d9',
                borderRadius: '6px',
                backgroundColor: '#000'
              }}
            />

            {isLoaded && (
              <div style={{ marginTop: 16, textAlign: 'center' }}>
                <Button.Group>
                  <Button icon={<ZoomInOutlined />} onClick={() => handleZoomChange(zoom * 1.2)}>
                    Zoom In
                  </Button>
                  <Button icon={<ZoomOutOutlined />} onClick={() => handleZoomChange(zoom / 1.2)}>
                    Zoom Out
                  </Button>
                  <Button icon={<RotateLeftOutlined />} onClick={() => handleRotate('left')}>
                    Rotate Left
                  </Button>
                  <Button icon={<RotateRightOutlined />} onClick={() => handleRotate('right')}>
                    Rotate Right
                  </Button>
                  <Button onClick={resetView}>
                    Reset View
                  </Button>
                </Button.Group>
              </div>
            )}
          </Card>
        </Col>

        <Col xs={24} lg={6}>
          <Card title="Viewer Controls" style={{ height: '100%' }}>
            {isLoaded ? (
              <div>
                <div style={{ marginBottom: 24 }}>
                  <h4>Window/Level</h4>
                  <div style={{ marginBottom: 16 }}>
                    <label>Window Width: {windowWidth}</label>
                    <Slider
                      min={1}
                      max={2000}
                      value={windowWidth}
                      onChange={handleWindowWidthChange}
                      style={{ marginTop: 8 }}
                    />
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <label>Window Center: {windowCenter}</label>
                    <Slider
                      min={-1000}
                      max={1000}
                      value={windowCenter}
                      onChange={handleWindowCenterChange}
                      style={{ marginTop: 8 }}
                    />
                  </div>
                </div>

                <div style={{ marginBottom: 24 }}>
                  <h4>Zoom</h4>
                  <div style={{ marginBottom: 16 }}>
                    <label>Scale: {zoom.toFixed(2)}x</label>
                    <Slider
                      min={0.1}
                      max={5}
                      step={0.1}
                      value={zoom}
                      onChange={handleZoomChange}
                      style={{ marginTop: 8 }}
                    />
                  </div>
                </div>

                <div style={{ marginBottom: 24 }}>
                  <h4>Rotation</h4>
                  <div style={{ marginBottom: 16 }}>
                    <label>Angle: {rotation}°</label>
                    <Slider
                      min={-180}
                      max={180}
                      value={rotation}
                      onChange={(value) => {
                        setRotation(value);
                        if (isLoaded && imageId) {
                          const element = canvasRef.current;
                          const viewport = cornerstone.getViewport(element);
                          viewport.rotation = value;
                          cornerstone.setViewport(element, viewport);
                        }
                      }}
                      style={{ marginTop: 8 }}
                    />
                  </div>
                </div>

                <div>
                  <h4>Tools</h4>
                  <p style={{ fontSize: 12, color: '#666' }}>
                    <strong>Left Click:</strong> Window/Level<br/>
                    <strong>Right Click:</strong> Zoom<br/>
                    <strong>Middle Click:</strong> Pan<br/>
                    <strong>Wheel:</strong> Zoom
                  </p>
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '40px 0', color: '#666' }}>
                <p>Upload a DICOM file to view it here</p>
                <p style={{ fontSize: 12 }}>
                  The viewer supports standard DICOM medical imaging files
                </p>
              </div>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default DICOMViewer;
