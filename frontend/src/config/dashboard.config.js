/**
 * Enterprise Dashboard Configuration
 * Real-time monitoring, analytics, and control panel
 */

// API Configuration
export const API_CONFIG = {
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v2',
  wsURL: import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws',
  timeout: 30000,
  retryAttempts: 3
};

// WebSocket Configuration
export const WS_CONFIG = {
  reconnectInterval: 3000,
  maxReconnectAttempts: 10,
  pingInterval: 30000,
  videoStreamFPS: 15
};

// Dashboard Modules
export const DASHBOARD_MODULES = {
  LIVE_MONITORING: {
    enabled: true,
    refreshRate: 2000, // 2 seconds
    maxBufferSize: 100
  },
  ANALYTICS: {
    enabled: true,
    timeWindows: ['1h', '24h', '7d', '30d'],
    defaultWindow: '24h'
  },
  DRIFT_MONITORING: {
    enabled: true,
    checkInterval: 60000, // 1 minute
    alertThresholds: {
      performance: 0.03, // 3% drop
      dataShift: 0.05, // 5% KS statistic
      conceptDrift: 0.1
    }
  },
  ANNOTATION_QUEUE: {
    enabled: true,
    pageSize: 20,
    autoRefresh: true,
    refreshInterval: 10000
  },
  TRACEABILITY: {
    enabled: true,
    qrCodeScanner: true,
    blockchainVerification: true
  },
  MODEL_PERFORMANCE: {
    enabled: true,
    models: [
      { id: 'yolov3', name: 'YOLOv3', type: 'detection' },
      { id: 'yolo_nas', name: 'YOLO-NAS', type: 'detection' },
      { id: 'resnet50', name: 'ResNet-50', type: 'classification' },
      { id: 'vgg19', name: 'VGG-19', type: 'classification' },
      { id: 'mobilenet_v2', name: 'MobileNetV2', type: 'classification' },
      { id: 'inception_v3', name: 'InceptionV3', type: 'classification' }
    ]
  }
};

// User Roles
export const USER_ROLES = {
  FARMER: {
    name: 'Farmer',
    permissions: ['view_predictions', 'create_batch', 'view_traceability']
  },
  QA_INSPECTOR: {
    name: 'QA Inspector',
    permissions: ['view_predictions', 'annotate', 'view_analytics', 'view_compliance']
  },
  PLANT_MANAGER: {
    name: 'Plant Manager',
    permissions: ['view_all', 'configure_system', 'view_analytics', 'export_reports']
  },
  ADMIN: {
    name: 'Administrator',
    permissions: ['*']
  }
};

// Fruit Categories
export const FRUIT_CATEGORIES = {
  apple: { label: 'Apple', color: '#FF6B6B' },
  banana: { label: 'Banana', color: '#FFD93D' },
  grape: { label: 'Grape', color: '#6BCB77' },
  orange: { label: 'Orange', color: '#FF8C42' },
  mango: { label: 'Mango', color: '#FFA500' },
  strawberry: { label: 'Strawberry', color: '#FF69B4' }
};

// Ripeness Levels
export const RIPENESS_LEVELS = {
  unripe: { label: 'Unripe', color: '#90EE90', shelfLife: 14 },
  ripe: { label: 'Ripe', color: '#FFD700', shelfLife: 7 },
  overripe: { label: 'Overripe', color: '#FF6347', shelfLife: 2 }
};

// Quality Grades
export const QUALITY_GRADES = {
  A: { label: 'Grade A', color: '#4CAF50', price: 1.5 },
  B: { label: 'Grade B', color: '#FFC107', price: 1.0 },
  C: { label: 'Grade C', color: '#FF9800', price: 0.7 },
  REJECT: { label: 'Reject', color: '#F44336', price: 0.0 }
};

// Defect Types
export const DEFECT_TYPES = [
  { id: 'bruise', label: 'Bruise', severity: 'medium' },
  { id: 'rot', label: 'Rot', severity: 'high' },
  { id: 'insect_damage', label: 'Insect Damage', severity: 'low' },
  { id: 'deformation', label: 'Deformation', severity: 'low' },
  { id: 'discoloration', label: 'Discoloration', severity: 'low' }
];

// Chart Configuration
export const CHART_CONFIG = {
  colors: {
    primary: '#3B82F6',
    success: '#10B981',
    warning: '#F59E0B',
    danger: '#EF4444',
    info: '#6366F1'
  },
  animation: {
    duration: 750,
    easing: 'easeInOutQuart'
  },
  responsive: true,
  maintainAspectRatio: false
};

// Performance Targets
export const PERFORMANCE_TARGETS = {
  detection_accuracy: 0.95,
  classification_accuracy: 0.93,
  ripeness_accuracy: 0.93,
  edge_latency_ms: 50,
  throughput_fps: 20,
  drift_threshold: 0.03
};

// Alert Severities
export const ALERT_SEVERITIES = {
  LOW: { color: '#10B981', icon: 'info' },
  MEDIUM: { color: '#F59E0B', icon: 'warning' },
  HIGH: { color: '#F97316', icon: 'alert' },
  CRITICAL: { color: '#EF4444', icon: 'error' }
};

// Compliance Standards
export const COMPLIANCE_STANDARDS = {
  FSSAI: {
    name: 'Food Safety and Standards Authority of India',
    requirements: ['traceability', 'quality_grade', 'shelf_life']
  },
  HACCP: {
    name: 'Hazard Analysis Critical Control Points',
    requirements: ['contamination_check', 'temperature_monitoring']
  },
  ISO22000: {
    name: 'ISO 22000 Food Safety Management',
    requirements: ['quality_management', 'traceability', 'safety']
  },
  GLOBALGAP: {
    name: 'GlobalGAP Good Agricultural Practices',
    requirements: ['origin_verification', 'pesticide_residues']
  }
};

// Export Data Formats
export const EXPORT_FORMATS = {
  CSV: { extension: 'csv', mimeType: 'text/csv' },
  JSON: { extension: 'json', mimeType: 'application/json' },
  PDF: { extension: 'pdf', mimeType: 'application/pdf' },
  EXCEL: { extension: 'xlsx', mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }
};

// Time Windows for Analytics
export const TIME_WINDOWS = [
  { value: '1h', label: 'Last Hour', seconds: 3600 },
  { value: '24h', label: 'Last 24 Hours', seconds: 86400 },
  { value: '7d', label: 'Last 7 Days', seconds: 604800 },
  { value: '30d', label: 'Last 30 Days', seconds: 2592000 }
];

// Animation Presets (GSAP)
export const ANIMATION_PRESETS = {
  fadeIn: {
    from: { opacity: 0, y: 20 },
    to: { opacity: 1, y: 0, duration: 0.6, ease: 'power3.out' }
  },
  slideIn: {
    from: { x: -50, opacity: 0 },
    to: { x: 0, opacity: 1, duration: 0.8, ease: 'power3.out' }
  },
  scaleIn: {
    from: { scale: 0.8, opacity: 0 },
    to: { scale: 1, opacity: 1, duration: 0.5, ease: 'back.out(1.7)' }
  },
  pulse: {
    to: { scale: 1.05, duration: 0.3, yoyo: true, repeat: 1 }
  }
};

// Notification Types
export const NOTIFICATION_TYPES = {
  SUCCESS: { icon: '✅', color: '#10B981' },
  ERROR: { icon: '❌', color: '#EF4444' },
  WARNING: { icon: '⚠️', color: '#F59E0B' },
  INFO: { icon: 'ℹ️', color: '#3B82F6' }
};

// Dashboard Layouts
export const DASHBOARD_LAYOUTS = {
  OPERATOR: [
    { id: 'live-feed', x: 0, y: 0, w: 8, h: 6 },
    { id: 'throughput', x: 8, y: 0, w: 4, h: 3 },
    { id: 'grade-distribution', x: 8, y: 3, w: 4, h: 3 }
  ],
  MANAGER: [
    { id: 'analytics', x: 0, y: 0, w: 6, h: 4 },
    { id: 'model-performance', x: 6, y: 0, w: 6, h: 4 },
    { id: 'drift-monitor', x: 0, y: 4, w: 12, h: 3 }
  ],
  QA: [
    { id: 'annotation-queue', x: 0, y: 0, w: 6, h: 8 },
    { id: 'quality-metrics', x: 6, y: 0, w: 6, h: 8 }
  ]
};

export default {
  API_CONFIG,
  WS_CONFIG,
  DASHBOARD_MODULES,
  USER_ROLES,
  FRUIT_CATEGORIES,
  RIPENESS_LEVELS,
  QUALITY_GRADES,
  DEFECT_TYPES,
  CHART_CONFIG,
  PERFORMANCE_TARGETS,
  ALERT_SEVERITIES,
  COMPLIANCE_STANDARDS,
  EXPORT_FORMATS,
  TIME_WINDOWS,
  ANIMATION_PRESETS,
  NOTIFICATION_TYPES,
  DASHBOARD_LAYOUTS
};
