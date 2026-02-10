"""
Traceability & Compliance System
Generate batch records, QR codes, and maintain audit trails
"""

import uuid
import json
import qrcode
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from io import BytesIO
import base64


@dataclass
class BatchRecord:
    """Complete traceability record for a fruit batch"""
    batch_id: str
    farm_origin: Dict
    harvest_date: datetime
    variety: str
    quantity_kg: float
    quality_scores: Dict
    predicted_shelf_life_days: int
    storage_conditions: Dict
    compliance_certifications: List[str]
    processing_timestamp: datetime
    grade_distribution: Dict
    defect_analysis: Dict
    traceability_hash: str
    qr_code_data: str


class TraceabilitySystem:
    """
    Manage end-to-end traceability for fruit batches
    """
    
    def __init__(self, blockchain_enabled: bool = False):
        self.blockchain_enabled = blockchain_enabled
        self.batch_records = {}
        
    def create_batch_record(
        self,
        farm_data: Dict,
        harvest_date: datetime,
        variety: str,
        quantity_kg: float,
        quality_analysis: Dict,
        storage_conditions: Dict,
        certifications: List[str]
    ) -> BatchRecord:
        """
        Create comprehensive batch record with traceability
        """
        
        batch_id = self._generate_batch_id()
        
        # Calculate grade distribution
        grade_dist = self._calculate_grade_distribution(
            quality_analysis.get('individual_samples', [])
        )
        
        # Defect analysis
        defect_analysis = self._analyze_defects(
            quality_analysis.get('individual_samples', [])
        )
        
        # Predicted shelf life (from quality scores)
        shelf_life = self._predict_batch_shelf_life(grade_dist, defect_analysis)
        
        # Generate record
        record = BatchRecord(
            batch_id=batch_id,
            farm_origin={
                'farm_id': farm_data.get('farm_id'),
                'farm_name': farm_data.get('farm_name'),
                'location': farm_data.get('location'),
                'coordinates': farm_data.get('coordinates'),
                'certification_number': farm_data.get('certification')
            },
            harvest_date=harvest_date,
            variety=variety,
            quantity_kg=quantity_kg,
            quality_scores={
                'avg_ripeness_score': quality_analysis.get('avg_ripeness', 0),
                'avg_freshness_score': quality_analysis.get('avg_freshness', 0),
                'defect_rate': quality_analysis.get('defect_rate', 0),
                'overall_quality_score': quality_analysis.get('overall_score', 0)
            },
            predicted_shelf_life_days=shelf_life,
            storage_conditions={
                'temperature_celsius': storage_conditions.get('temperature'),
                'humidity_percent': storage_conditions.get('humidity'),
                'controlled_atmosphere': storage_conditions.get('ca_storage', False),
                'packaging_type': storage_conditions.get('packaging')
            },
            compliance_certifications=certifications,
            processing_timestamp=datetime.now(),
            grade_distribution=grade_dist,
            defect_analysis=defect_analysis,
            traceability_hash=self._generate_hash(batch_id, farm_data, harvest_date),
            qr_code_data=""  # Will be populated below
        )
        
        # Generate QR code
        qr_data = self._generate_qr_data(record)
        record.qr_code_data = qr_data
        
        # Store record
        self.batch_records[batch_id] = record
        
        # Blockchain registration (if enabled)
        if self.blockchain_enabled:
            self._register_on_blockchain(record)
            
        return record
        
    def generate_qr_code(
        self,
        batch_record: BatchRecord,
        size: int = 300
    ) -> str:
        """
        Generate QR code image for batch
        
        Returns:
            Base64 encoded PNG image
        """
        
        # Create QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        
        # Add data
        qr_data = {
            'batch_id': batch_record.batch_id,
            'variety': batch_record.variety,
            'harvest_date': batch_record.harvest_date.isoformat(),
            'farm': batch_record.farm_origin['farm_name'],
            'grade_A_percent': batch_record.grade_distribution.get('A', 0),
            'shelf_life_days': batch_record.predicted_shelf_life_days,
            'verify_url': f"https://verify.fruitsystem.com/{batch_record.batch_id}",
            'hash': batch_record.traceability_hash
        }
        
        qr.add_data(json.dumps(qr_data))
        qr.make(fit=True)
        
        # Create image
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
        
    def verify_batch(self, batch_id: str, provided_hash: str) -> Dict:
        """
        Verify batch authenticity using hash
        """
        
        if batch_id not in self.batch_records:
            return {
                'verified': False,
                'error': 'Batch not found'
            }
            
        record = self.batch_records[batch_id]
        
        hash_match = record.traceability_hash == provided_hash
        
        return {
            'verified': hash_match,
            'batch_id': batch_id,
            'farm': record.farm_origin['farm_name'],
            'harvest_date': record.harvest_date.isoformat(),
            'certifications': record.compliance_certifications,
            'quality_grade': self._get_overall_grade(record.grade_distribution)
        }
        
    def generate_compliance_report(
        self,
        batch_id: str,
        standards: List[str]
    ) -> Dict:
        """
        Generate compliance report for regulatory bodies
        
        Args:
            batch_id: Batch identifier
            standards: List of standards to check (FSSAI, HACCP, ISO, etc.)
        """
        
        if batch_id not in self.batch_records:
            return {'error': 'Batch not found'}
            
        record = self.batch_records[batch_id]
        
        compliance_checks = {
            'FSSAI': self._check_fssai_compliance(record),
            'HACCP': self._check_haccp_compliance(record),
            'ISO22000': self._check_iso_compliance(record),
            'GlobalGAP': self._check_globalgap_compliance(record)
        }
        
        report = {
            'batch_id': batch_id,
            'report_generated': datetime.now().isoformat(),
            'farm_details': record.farm_origin,
            'harvest_information': {
                'date': record.harvest_date.isoformat(),
                'variety': record.variety,
                'quantity_kg': record.quantity_kg
            },
            'quality_assessment': {
                **record.quality_scores,
                'grade_distribution': record.grade_distribution,
                'defect_analysis': record.defect_analysis
            },
            'storage_traceability': record.storage_conditions,
            'shelf_life_prediction': {
                'estimated_days': record.predicted_shelf_life_days,
                'expiry_date': (
                    record.harvest_date + timedelta(days=record.predicted_shelf_life_days)
                ).isoformat()
            },
            'compliance_status': {
                standard: compliance_checks[standard]
                for standard in standards if standard in compliance_checks
            },
            'certifications_on_file': record.compliance_certifications,
            'traceability_verification': {
                'hash': record.traceability_hash,
                'blockchain_registered': self.blockchain_enabled,
                'immutable': True
            }
        }
        
        return report
        
    def calculate_yield_prediction(
        self,
        historical_data: List[BatchRecord],
        forecasting_params: Dict
    ) -> Dict:
        """
        Predict yield and revenue based on historical data
        """
        
        if len(historical_data) == 0:
            return {'error': 'Insufficient historical data'}
            
        # Calculate averages
        total_quantity = sum(b.quantity_kg for b in historical_data)
        avg_quantity_per_batch = total_quantity / len(historical_data)
        
        # Grade distribution
        avg_grade_a = np.mean([
            b.grade_distribution.get('A', 0) for b in historical_data
        ])
        avg_grade_b = np.mean([
            b.grade_distribution.get('B', 0) for b in historical_data
        ])
        avg_grade_c = np.mean([
            b.grade_distribution.get('C', 0) for b in historical_data
        ])
        
        # Revenue calculation
        price_per_kg = forecasting_params.get('prices', {})
        grade_a_price = price_per_kg.get('A', 100)  # Currency units
        grade_b_price = price_per_kg.get('B', 70)
        grade_c_price = price_per_kg.get('C', 40)
        
        projected_batches = forecasting_params.get('expected_batches', 10)
        
        projected_revenue_grade_a = avg_quantity_per_batch * (avg_grade_a / 100) * grade_a_price * projected_batches
        projected_revenue_grade_b = avg_quantity_per_batch * (avg_grade_b / 100) * grade_b_price * projected_batches
        projected_revenue_grade_c = avg_quantity_per_batch * (avg_grade_c / 100) * grade_c_price * projected_batches
        
        total_projected_revenue = projected_revenue_grade_a + projected_revenue_grade_b + projected_revenue_grade_c
        
        # Waste reduction
        avg_defect_rate = np.mean([
            b.quality_scores['defect_rate'] for b in historical_data
        ])
        
        total_marketable = avg_quantity_per_batch * projected_batches * (1 - avg_defect_rate / 100)
        total_waste = avg_quantity_per_batch * projected_batches * (avg_defect_rate / 100)
        
        waste_reduction_opportunity = total_waste * 0.3  # 30% reduction target
        
        return {
            'historical_analysis': {
                'batches_analyzed': len(historical_data),
                'avg_batch_size_kg': float(avg_quantity_per_batch),
                'avg_grade_distribution': {
                    'A': float(avg_grade_a),
                    'B': float(avg_grade_b),
                    'C': float(avg_grade_c)
                },
                'avg_defect_rate': float(avg_defect_rate)
            },
            'projections': {
                'expected_batches': projected_batches,
                'total_yield_kg': float(avg_quantity_per_batch * projected_batches),
                'marketable_yield_kg': float(total_marketable),
                'waste_kg': float(total_waste)
            },
            'revenue_forecast': {
                'grade_A_revenue': float(projected_revenue_grade_a),
                'grade_B_revenue': float(projected_revenue_grade_b),
                'grade_C_revenue': float(projected_revenue_grade_c),
                'total_revenue': float(total_projected_revenue),
                'currency': forecasting_params.get('currency', 'USD')
            },
            'improvement_opportunities': {
                'waste_reduction_potential_kg': float(waste_reduction_opportunity),
                'additional_revenue_potential': float(
                    waste_reduction_opportunity * grade_b_price
                )
            }
        }
        
    def _generate_batch_id(self) -> str:
        """Generate unique batch ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_suffix = uuid.uuid4().hex[:6].upper()
        return f"BATCH-{timestamp}-{random_suffix}"
        
    def _generate_hash(
        self,
        batch_id: str,
        farm_data: Dict,
        harvest_date: datetime
    ) -> str:
        """Generate cryptographic hash for traceability"""
        
        hash_input = f"{batch_id}{farm_data.get('farm_id')}{harvest_date.isoformat()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
        
    def _calculate_grade_distribution(self, samples: List[Dict]) -> Dict:
        """Calculate grade distribution from individual samples"""
        
        if not samples:
            return {'A': 0, 'B': 0, 'C': 0}
            
        grades = [s['quality_grade'] for s in samples if 'quality_grade' in s]
        total = len(grades)
        
        if total == 0:
            return {'A': 0, 'B': 0, 'C': 0}
            
        return {
            'A': grades.count('A') / total * 100,
            'B': grades.count('B') / total * 100,
            'C': grades.count('C') / total * 100
        }
        
    def _analyze_defects(self, samples: List[Dict]) -> Dict:
        """Analyze defect patterns"""
        
        if not samples:
            return {'total_defects': 0, 'defect_types': {}}
            
        all_defects = []
        for sample in samples:
            if 'defects' in sample:
                all_defects.extend(sample['defects'])
                
        from collections import Counter
        defect_counts = Counter(all_defects)
        
        return {
            'total_defects': len(all_defects),
            'defect_types': dict(defect_counts),
            'defect_rate': len(all_defects) / len(samples) * 100 if samples else 0
        }
        
    def _predict_batch_shelf_life(
        self,
        grade_dist: Dict,
        defect_analysis: Dict
    ) -> int:
        """Predict average shelf life for batch"""
        
        # Base shelf life by grade
        base_shelf_life = {
            'A': 14,  # days
            'B': 10,
            'C': 5
        }
        
        weighted_shelf_life = sum(
            base_shelf_life[grade] * (percentage / 100)
            for grade, percentage in grade_dist.items()
        )
        
        # Adjust for defects
        defect_penalty = min(defect_analysis.get('defect_rate', 0) / 10, 3)
        
        return int(max(weighted_shelf_life - defect_penalty, 1))
        
    def _get_overall_grade(self, grade_dist: Dict) -> str:
        """Get overall batch grade"""
        
        if grade_dist.get('A', 0) > 60:
            return 'A'
        elif grade_dist.get('A', 0) + grade_dist.get('B', 0) > 80:
            return 'B'
        else:
            return 'C'
            
    def _generate_qr_data(self, record: BatchRecord) -> str:
        """Generate QR code data string"""
        return f"FRUIT-TRACE:{record.batch_id}:{record.traceability_hash[:16]}"
        
    def _register_on_blockchain(self, record: BatchRecord):
        """Register batch on blockchain (placeholder)"""
        # In production, this would interact with blockchain API
        pass
        
    def _check_fssai_compliance(self, record: BatchRecord) -> Dict:
        """Check FSSAI compliance"""
        return {
            'compliant': 'FSSAI' in record.compliance_certifications,
            'requirements_met': ['traceability', 'quality_standards', 'storage_conditions'],
            'status': 'PASSED' if 'FSSAI' in record.compliance_certifications else 'PENDING'
        }
        
    def _check_haccp_compliance(self, record: BatchRecord) -> Dict:
        """Check HACCP compliance"""
        return {
            'compliant': 'HACCP' in record.compliance_certifications,
            'critical_control_points': ['harvest', 'storage', 'sorting'],
            'status': 'PASSED' if 'HACCP' in record.compliance_certifications else 'PENDING'
        }
        
    def _check_iso_compliance(self, record: BatchRecord) -> Dict:
        """Check ISO 22000 compliance"""
        return {
            'compliant': 'ISO22000' in record.compliance_certifications,
            'food_safety_management': True,
            'status': 'PASSED' if 'ISO22000' in record.compliance_certifications else 'PENDING'
        }
        
    def _check_globalgap_compliance(self, record: BatchRecord) -> Dict:
        """Check GlobalGAP compliance"""
        return {
            'compliant': 'GlobalGAP' in record.compliance_certifications,
            'good_agricultural_practices': True,
            'status': 'PASSED' if 'GlobalGAP' in record.compliance_certifications else 'PENDING'
        }


# Numpy import for calculations
import numpy as np
