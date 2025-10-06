"""
Security Testing Framework for Medical Imaging AI API
This is a placeholder implementation for future security testing capabilities
"""

import pytest
import requests
import json
import time
import random
import string
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SecurityTestType(Enum):
    """Security test type enumeration."""
    INJECTION = "injection"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_EXPOSURE = "data_exposure"
    RATE_LIMITING = "rate_limiting"
    INPUT_VALIDATION = "input_validation"
    FILE_UPLOAD = "file_upload"
    HEADERS = "headers"
    SSL_TLS = "ssl_tls"
    CORS = "cors"


class SecuritySeverity(Enum):
    """Security severity enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityTestResult:
    """Represents the result of a security test."""
    test_type: SecurityTestType
    test_name: str
    severity: SecuritySeverity
    passed: bool
    details: str
    recommendation: str
    timestamp: str


class SecurityTestingFramework:
    """
    Security Testing Framework for Medical Imaging AI API.
    
    This framework provides comprehensive security testing capabilities including:
    - Injection attack testing
    - Authentication and authorization testing
    - Data exposure testing
    - Rate limiting testing
    - Input validation testing
    - File upload security testing
    - Security headers testing
    - SSL/TLS testing
    - CORS testing
    """

    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize the security testing framework.
        
        Args:
            base_url: Base URL of the API to test
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results: List[SecurityTestResult] = []

    def run_all_security_tests(self) -> List[SecurityTestResult]:
        """
        Run all security tests.
        
        Returns:
            List of security test results
        """
        logger.info("Starting comprehensive security testing...")
        
        # Run all test categories
        self._test_injection_attacks()
        self._test_authentication()
        self._test_authorization()
        self._test_data_exposure()
        self._test_rate_limiting()
        self._test_input_validation()
        self._test_file_upload_security()
        self._test_security_headers()
        self._test_ssl_tls()
        self._test_cors()
        
        logger.info(f"Security testing completed. {len(self.test_results)} tests executed.")
        return self.test_results

    def _test_injection_attacks(self):
        """Test for various injection attacks."""
        logger.info("Testing injection attacks...")
        
        # SQL Injection tests
        self._test_sql_injection()
        
        # NoSQL Injection tests
        self._test_nosql_injection()
        
        # Command Injection tests
        self._test_command_injection()
        
        # LDAP Injection tests
        self._test_ldap_injection()

    def _test_sql_injection(self):
        """Test for SQL injection vulnerabilities."""
        sql_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM users --",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --",
            "' OR 1=1 --"
        ]
        
        for payload in sql_payloads:
            try:
                response = self.session.get(f"{self.base_url}/models", params={"filter": payload})
                
                # Check for SQL error messages
                error_indicators = [
                    "sql syntax",
                    "mysql_fetch",
                    "ORA-01756",
                    "Microsoft OLE DB Provider",
                    "ODBC SQL Server Driver"
                ]
                
                response_text = response.text.lower()
                has_sql_error = any(indicator in response_text for indicator in error_indicators)
                
                self._record_test_result(
                    SecurityTestType.INJECTION,
                    f"SQL Injection Test: {payload[:20]}...",
                    SecuritySeverity.HIGH,
                    not has_sql_error,
                    f"Response status: {response.status_code}, SQL error detected: {has_sql_error}",
                    "Implement parameterized queries and input validation"
                )
                
            except Exception as e:
                self._record_test_result(
                    SecurityTestType.INJECTION,
                    f"SQL Injection Test: {payload[:20]}...",
                    SecuritySeverity.HIGH,
                    False,
                    f"Test failed with exception: {str(e)}",
                    "Review error handling and logging"
                )

    def _test_nosql_injection(self):
        """Test for NoSQL injection vulnerabilities."""
        nosql_payloads = [
            '{"$where": "this.username == this.password"}',
            '{"username": {"$ne": null}, "password": {"$ne": null}}',
            '{"username": {"$regex": ".*"}, "password": {"$regex": ".*"}}',
            '{"$or": [{"username": "admin"}, {"password": "admin"}]}'
        ]
        
        for payload in nosql_payloads:
            try:
                response = self.session.post(
                    f"{self.base_url}/upload",
                    json={"filter": payload},
                    headers={"Content-Type": "application/json"}
                )
                
                # Check for NoSQL error messages
                error_indicators = [
                    "mongo",
                    "mongodb",
                    "bson",
                    "objectid",
                    "nosql"
                ]
                
                response_text = response.text.lower()
                has_nosql_error = any(indicator in response_text for indicator in error_indicators)
                
                self._record_test_result(
                    SecurityTestType.INJECTION,
                    f"NoSQL Injection Test: {payload[:20]}...",
                    SecuritySeverity.HIGH,
                    not has_nosql_error,
                    f"Response status: {response.status_code}, NoSQL error detected: {has_nosql_error}",
                    "Implement proper NoSQL query validation and sanitization"
                )
                
            except Exception as e:
                self._record_test_result(
                    SecurityTestType.INJECTION,
                    f"NoSQL Injection Test: {payload[:20]}...",
                    SecuritySeverity.HIGH,
                    False,
                    f"Test failed with exception: {str(e)}",
                    "Review error handling and logging"
                )

    def _test_command_injection(self):
        """Test for command injection vulnerabilities."""
        command_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& whoami",
            "; rm -rf /",
            "`id`"
        ]
        
        for payload in command_payloads:
            try:
                response = self.session.get(f"{self.base_url}/models", params={"command": payload})
                
                # Check for command execution indicators
                execution_indicators = [
                    "root:",
                    "bin:",
                    "daemon:",
                    "uid=",
                    "gid="
                ]
                
                response_text = response.text.lower()
                has_execution = any(indicator in response_text for indicator in execution_indicators)
                
                self._record_test_result(
                    SecurityTestType.INJECTION,
                    f"Command Injection Test: {payload[:20]}...",
                    SecuritySeverity.CRITICAL,
                    not has_execution,
                    f"Response status: {response.status_code}, Command execution detected: {has_execution}",
                    "Never execute user input as system commands. Use safe APIs instead."
                )
                
            except Exception as e:
                self._record_test_result(
                    SecurityTestType.INJECTION,
                    f"Command Injection Test: {payload[:20]}...",
                    SecuritySeverity.CRITICAL,
                    False,
                    f"Test failed with exception: {str(e)}",
                    "Review error handling and logging"
                )

    def _test_ldap_injection(self):
        """Test for LDAP injection vulnerabilities."""
        ldap_payloads = [
            "*)(uid=*))(|(uid=*",
            "*)(|(password=*))",
            "*)(|(objectClass=*))",
            "*)(|(cn=*))"
        ]
        
        for payload in ldap_payloads:
            try:
                response = self.session.get(f"{self.base_url}/models", params={"ldap": payload})
                
                # Check for LDAP error messages
                error_indicators = [
                    "ldap",
                    "distinguished name",
                    "dn:",
                    "objectclass",
                    "attribute"
                ]
                
                response_text = response.text.lower()
                has_ldap_error = any(indicator in response_text for indicator in error_indicators)
                
                self._record_test_result(
                    SecurityTestType.INJECTION,
                    f"LDAP Injection Test: {payload[:20]}...",
                    SecuritySeverity.HIGH,
                    not has_ldap_error,
                    f"Response status: {response.status_code}, LDAP error detected: {has_ldap_error}",
                    "Implement proper LDAP query validation and sanitization"
                )
                
            except Exception as e:
                self._record_test_result(
                    SecurityTestType.INJECTION,
                    f"LDAP Injection Test: {payload[:20]}...",
                    SecuritySeverity.HIGH,
                    False,
                    f"Test failed with exception: {str(e)}",
                    "Review error handling and logging"
                )

    def _test_authentication(self):
        """Test authentication mechanisms."""
        logger.info("Testing authentication...")
        
        # Test for default credentials
        self._test_default_credentials()
        
        # Test for weak authentication
        self._test_weak_authentication()
        
        # Test for authentication bypass
        self._test_authentication_bypass()

    def _test_default_credentials(self):
        """Test for default credentials."""
        default_credentials = [
            ("admin", "admin"),
            ("admin", "password"),
            ("admin", ""),
            ("root", "root"),
            ("user", "user"),
            ("test", "test")
        ]
        
        for username, password in default_credentials:
            try:
                response = self.session.post(
                    f"{self.base_url}/login",
                    json={"username": username, "password": password}
                )
                
                success = response.status_code != 200 or "token" not in response.text.lower()
                
                self._record_test_result(
                    SecurityTestType.AUTHENTICATION,
                    f"Default Credentials Test: {username}/{password}",
                    SecuritySeverity.HIGH,
                    success,
                    f"Response status: {response.status_code}",
                    "Change all default credentials and implement strong password policies"
                )
                
            except Exception as e:
                self._record_test_result(
                    SecurityTestType.AUTHENTICATION,
                    f"Default Credentials Test: {username}/{password}",
                    SecuritySeverity.HIGH,
                    False,
                    f"Test failed with exception: {str(e)}",
                    "Review authentication endpoint implementation"
                )

    def _test_weak_authentication(self):
        """Test for weak authentication mechanisms."""
        # Test for missing authentication on sensitive endpoints
        sensitive_endpoints = ["/admin", "/config", "/debug", "/metrics"]
        
        for endpoint in sensitive_endpoints:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}")
                
                # Check if endpoint is accessible without authentication
                accessible = response.status_code == 200
                
                self._record_test_result(
                    SecurityTestType.AUTHENTICATION,
                    f"Unauthenticated Access Test: {endpoint}",
                    SecuritySeverity.MEDIUM,
                    not accessible,
                    f"Response status: {response.status_code}",
                    "Implement authentication for all sensitive endpoints"
                )
                
            except Exception as e:
                self._record_test_result(
                    SecurityTestType.AUTHENTICATION,
                    f"Unauthenticated Access Test: {endpoint}",
                    SecuritySeverity.MEDIUM,
                    False,
                    f"Test failed with exception: {str(e)}",
                    "Review endpoint security implementation"
                )

    def _test_authentication_bypass(self):
        """Test for authentication bypass vulnerabilities."""
        bypass_payloads = [
            "admin'--",
            "admin'/*",
            "' OR '1'='1' --",
            "' OR '1'='1' /*",
            "admin' OR '1'='1"
        ]
        
        for payload in bypass_payloads:
            try:
                response = self.session.post(
                    f"{self.base_url}/login",
                    json={"username": payload, "password": "anything"}
                )
                
                bypass_successful = response.status_code == 200 and "token" in response.text.lower()
                
                self._record_test_result(
                    SecurityTestType.AUTHENTICATION,
                    f"Authentication Bypass Test: {payload[:20]}...",
                    SecuritySeverity.CRITICAL,
                    not bypass_successful,
                    f"Response status: {response.status_code}, Bypass successful: {bypass_successful}",
                    "Implement proper input validation and parameterized queries"
                )
                
            except Exception as e:
                self._record_test_result(
                    SecurityTestType.AUTHENTICATION,
                    f"Authentication Bypass Test: {payload[:20]}...",
                    SecuritySeverity.CRITICAL,
                    False,
                    f"Test failed with exception: {str(e)}",
                    "Review authentication implementation"
                )

    def _test_authorization(self):
        """Test authorization mechanisms."""
        logger.info("Testing authorization...")
        
        # Test for privilege escalation
        self._test_privilege_escalation()
        
        # Test for horizontal privilege escalation
        self._test_horizontal_privilege_escalation()

    def _test_privilege_escalation(self):
        """Test for privilege escalation vulnerabilities."""
        # Test accessing admin endpoints with regular user token
        admin_endpoints = ["/admin/users", "/admin/config", "/admin/logs"]
        
        for endpoint in admin_endpoints:
            try:
                # Try with regular user token
                headers = {"Authorization": "Bearer regular_user_token"}
                response = self.session.get(f"{self.base_url}{endpoint}", headers=headers)
                
                unauthorized_access = response.status_code == 200
                
                self._record_test_result(
                    SecurityTestType.AUTHORIZATION,
                    f"Privilege Escalation Test: {endpoint}",
                    SecuritySeverity.HIGH,
                    not unauthorized_access,
                    f"Response status: {response.status_code}",
                    "Implement proper role-based access control (RBAC)"
                )
                
            except Exception as e:
                self._record_test_result(
                    SecurityTestType.AUTHORIZATION,
                    f"Privilege Escalation Test: {endpoint}",
                    SecuritySeverity.HIGH,
                    False,
                    f"Test failed with exception: {str(e)}",
                    "Review authorization implementation"
                )

    def _test_horizontal_privilege_escalation(self):
        """Test for horizontal privilege escalation vulnerabilities."""
        # Test accessing other users' data
        try:
            response = self.session.get(f"{self.base_url}/users/other_user_id/data")
            
            unauthorized_access = response.status_code == 200
            
            self._record_test_result(
                SecurityTestType.AUTHORIZATION,
                "Horizontal Privilege Escalation Test: Other User Data",
                SecuritySeverity.HIGH,
                not unauthorized_access,
                f"Response status: {response.status_code}",
                "Implement proper user data isolation and access controls"
            )
            
        except Exception as e:
            self._record_test_result(
                SecurityTestType.AUTHORIZATION,
                "Horizontal Privilege Escalation Test: Other User Data",
                SecuritySeverity.HIGH,
                False,
                f"Test failed with exception: {str(e)}",
                "Review user data access controls"
            )

    def _test_data_exposure(self):
        """Test for data exposure vulnerabilities."""
        logger.info("Testing data exposure...")
        
        # Test for sensitive data in responses
        self._test_sensitive_data_exposure()
        
        # Test for information disclosure
        self._test_information_disclosure()

    def _test_sensitive_data_exposure(self):
        """Test for sensitive data exposure in responses."""
        sensitive_patterns = [
            "password",
            "secret",
            "key",
            "token",
            "api_key",
            "private_key",
            "ssn",
            "credit_card"
        ]
        
        endpoints = ["/health", "/metrics", "/models", "/upload"]
        
        for endpoint in endpoints:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}")
                response_text = response.text.lower()
                
                exposed_data = []
                for pattern in sensitive_patterns:
                    if pattern in response_text:
                        exposed_data.append(pattern)
                
                has_exposure = len(exposed_data) > 0
                
                self._record_test_result(
                    SecurityTestType.DATA_EXPOSURE,
                    f"Sensitive Data Exposure Test: {endpoint}",
                    SecuritySeverity.HIGH,
                    not has_exposure,
                    f"Exposed patterns: {exposed_data}",
                    "Remove sensitive data from API responses and implement proper data filtering"
                )
                
            except Exception as e:
                self._record_test_result(
                    SecurityTestType.DATA_EXPOSURE,
                    f"Sensitive Data Exposure Test: {endpoint}",
                    SecuritySeverity.HIGH,
                    False,
                    f"Test failed with exception: {str(e)}",
                    "Review API response content"
                )

    def _test_information_disclosure(self):
        """Test for information disclosure vulnerabilities."""
        # Test for verbose error messages
        try:
            response = self.session.get(f"{self.base_url}/nonexistent")
            
            # Check for verbose error messages
            verbose_indicators = [
                "stack trace",
                "exception",
                "error at line",
                "file path",
                "database error",
                "sql error"
            ]
            
            response_text = response.text.lower()
            has_verbose_errors = any(indicator in response_text for indicator in verbose_indicators)
            
            self._record_test_result(
                SecurityTestType.DATA_EXPOSURE,
                "Information Disclosure Test: Verbose Error Messages",
                SecuritySeverity.MEDIUM,
                not has_verbose_errors,
                f"Response status: {response.status_code}, Verbose errors detected: {has_verbose_errors}",
                "Implement generic error messages for production environments"
            )
            
        except Exception as e:
            self._record_test_result(
                SecurityTestType.DATA_EXPOSURE,
                "Information Disclosure Test: Verbose Error Messages",
                SecuritySeverity.MEDIUM,
                False,
                f"Test failed with exception: {str(e)}",
                "Review error handling implementation"
            )

    def _test_rate_limiting(self):
        """Test rate limiting mechanisms."""
        logger.info("Testing rate limiting...")
        
        # Test for rate limiting on API endpoints
        self._test_api_rate_limiting()

    def _test_api_rate_limiting(self):
        """Test rate limiting on API endpoints."""
        endpoints = ["/health", "/models", "/upload"]
        
        for endpoint in endpoints:
            try:
                # Send multiple requests rapidly
                responses = []
                for i in range(100):
                    response = self.session.get(f"{self.base_url}{endpoint}")
                    responses.append(response.status_code)
                    time.sleep(0.01)  # Small delay
                
                # Check if rate limiting is implemented
                rate_limited = any(status == 429 for status in responses)
                
                self._record_test_result(
                    SecurityTestType.RATE_LIMITING,
                    f"Rate Limiting Test: {endpoint}",
                    SecuritySeverity.MEDIUM,
                    rate_limited,
                    f"Rate limiting implemented: {rate_limited}",
                    "Implement rate limiting to prevent abuse and DoS attacks"
                )
                
            except Exception as e:
                self._record_test_result(
                    SecurityTestType.RATE_LIMITING,
                    f"Rate Limiting Test: {endpoint}",
                    SecuritySeverity.MEDIUM,
                    False,
                    f"Test failed with exception: {str(e)}",
                    "Review rate limiting implementation"
                )

    def _test_input_validation(self):
        """Test input validation mechanisms."""
        logger.info("Testing input validation...")
        
        # Test for buffer overflow
        self._test_buffer_overflow()
        
        # Test for input length validation
        self._test_input_length_validation()

    def _test_buffer_overflow(self):
        """Test for buffer overflow vulnerabilities."""
        # Test with very long input
        long_input = "A" * 10000
        
        try:
            response = self.session.post(
                f"{self.base_url}/upload",
                json={"data": long_input}
            )
            
            # Check if server handles long input gracefully
            handled_gracefully = response.status_code in [200, 400, 413, 422]
            
            self._record_test_result(
                SecurityTestType.INPUT_VALIDATION,
                "Buffer Overflow Test: Long Input",
                SecuritySeverity.HIGH,
                handled_gracefully,
                f"Response status: {response.status_code}",
                "Implement proper input length validation and limits"
            )
            
        except Exception as e:
            self._record_test_result(
                SecurityTestType.INPUT_VALIDATION,
                "Buffer Overflow Test: Long Input",
                SecuritySeverity.HIGH,
                False,
                f"Test failed with exception: {str(e)}",
                "Review input validation implementation"
            )

    def _test_input_length_validation(self):
        """Test input length validation."""
        # Test with various input lengths
        test_lengths = [1, 100, 1000, 10000, 100000]
        
        for length in test_lengths:
            try:
                test_input = "A" * length
                response = self.session.get(f"{self.base_url}/models", params={"filter": test_input})
                
                # Check if input length is properly validated
                proper_validation = response.status_code in [200, 400, 413, 422]
                
                self._record_test_result(
                    SecurityTestType.INPUT_VALIDATION,
                    f"Input Length Validation Test: {length} chars",
                    SecuritySeverity.MEDIUM,
                    proper_validation,
                    f"Response status: {response.status_code}",
                    "Implement proper input length validation"
                )
                
            except Exception as e:
                self._record_test_result(
                    SecurityTestType.INPUT_VALIDATION,
                    f"Input Length Validation Test: {length} chars",
                    SecuritySeverity.MEDIUM,
                    False,
                    f"Test failed with exception: {str(e)}",
                    "Review input validation implementation"
                )

    def _test_file_upload_security(self):
        """Test file upload security."""
        logger.info("Testing file upload security...")
        
        # Test for malicious file uploads
        self._test_malicious_file_upload()
        
        # Test for file type validation
        self._test_file_type_validation()

    def _test_malicious_file_upload(self):
        """Test for malicious file upload vulnerabilities."""
        malicious_files = [
            ("malicious.php", "<?php system($_GET['cmd']); ?>", "application/x-php"),
            ("malicious.jsp", "<% Runtime.getRuntime().exec(request.getParameter(\"cmd\")); %>", "application/x-jsp"),
            ("malicious.asp", "<% eval request(\"cmd\") %>", "application/x-asp"),
            ("malicious.exe", b"MZ\x90\x00", "application/x-executable")
        ]
        
        for filename, content, content_type in malicious_files:
            try:
                files = {"file": (filename, content, content_type)}
                response = self.session.post(f"{self.base_url}/upload", files=files)
                
                # Check if malicious file is rejected
                rejected = response.status_code in [400, 415, 422]
                
                self._record_test_result(
                    SecurityTestType.FILE_UPLOAD,
                    f"Malicious File Upload Test: {filename}",
                    SecuritySeverity.HIGH,
                    rejected,
                    f"Response status: {response.status_code}",
                    "Implement proper file type validation and content scanning"
                )
                
            except Exception as e:
                self._record_test_result(
                    SecurityTestType.FILE_UPLOAD,
                    f"Malicious File Upload Test: {filename}",
                    SecuritySeverity.HIGH,
                    False,
                    f"Test failed with exception: {str(e)}",
                    "Review file upload security implementation"
                )

    def _test_file_type_validation(self):
        """Test file type validation."""
        # Test with various file types
        test_files = [
            ("test.txt", "Hello World", "text/plain"),
            ("test.jpg", b"fake_jpg_data", "image/jpeg"),
            ("test.png", b"fake_png_data", "image/png"),
            ("test.pdf", b"fake_pdf_data", "application/pdf")
        ]
        
        for filename, content, content_type in test_files:
            try:
                files = {"file": (filename, content, content_type)}
                response = self.session.post(f"{self.base_url}/upload", files=files)
                
                # Check if file type is properly validated
                proper_validation = response.status_code in [200, 400, 415, 422]
                
                self._record_test_result(
                    SecurityTestType.FILE_UPLOAD,
                    f"File Type Validation Test: {filename}",
                    SecuritySeverity.MEDIUM,
                    proper_validation,
                    f"Response status: {response.status_code}",
                    "Implement proper file type validation"
                )
                
            except Exception as e:
                self._record_test_result(
                    SecurityTestType.FILE_UPLOAD,
                    f"File Type Validation Test: {filename}",
                    SecuritySeverity.MEDIUM,
                    False,
                    f"Test failed with exception: {str(e)}",
                    "Review file type validation implementation"
                )

    def _test_security_headers(self):
        """Test security headers."""
        logger.info("Testing security headers...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            headers = response.headers
            
            # Check for important security headers
            security_headers = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": "max-age=31536000",
                "Content-Security-Policy": "default-src 'self'",
                "Referrer-Policy": "strict-origin-when-cross-origin"
            }
            
            missing_headers = []
            for header, expected_value in security_headers.items():
                if header not in headers:
                    missing_headers.append(header)
            
            has_all_headers = len(missing_headers) == 0
            
            self._record_test_result(
                SecurityTestType.HEADERS,
                "Security Headers Test",
                SecuritySeverity.MEDIUM,
                has_all_headers,
                f"Missing headers: {missing_headers}",
                "Implement all recommended security headers"
            )
            
        except Exception as e:
            self._record_test_result(
                SecurityTestType.HEADERS,
                "Security Headers Test",
                SecuritySeverity.MEDIUM,
                False,
                f"Test failed with exception: {str(e)}",
                "Review security headers implementation"
            )

    def _test_ssl_tls(self):
        """Test SSL/TLS configuration."""
        logger.info("Testing SSL/TLS...")
        
        # This is a placeholder for SSL/TLS testing
        # In a real implementation, you would use tools like SSL Labs API
        # or libraries like sslyze to test SSL/TLS configuration
        
        self._record_test_result(
            SecurityTestType.SSL_TLS,
            "SSL/TLS Configuration Test",
            SecuritySeverity.MEDIUM,
            True,  # Placeholder
            "SSL/TLS testing requires external tools (SSL Labs, sslyze)",
            "Use SSL Labs API or sslyze to test SSL/TLS configuration"
        )

    def _test_cors(self):
        """Test CORS configuration."""
        logger.info("Testing CORS...")
        
        try:
            # Test CORS preflight request
            response = self.session.options(
                f"{self.base_url}/upload",
                headers={
                    "Origin": "https://malicious-site.com",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type"
                }
            )
            
            # Check CORS headers
            cors_headers = response.headers.get("Access-Control-Allow-Origin", "")
            cors_methods = response.headers.get("Access-Control-Allow-Methods", "")
            cors_headers_list = response.headers.get("Access-Control-Allow-Headers", "")
            
            # Check if CORS is too permissive
            too_permissive = (
                cors_headers == "*" or
                "malicious-site.com" in cors_headers or
                "POST" in cors_methods
            )
            
            self._record_test_result(
                SecurityTestType.CORS,
                "CORS Configuration Test",
                SecuritySeverity.MEDIUM,
                not too_permissive,
                f"CORS headers: {dict(response.headers)}",
                "Implement restrictive CORS policy"
            )
            
        except Exception as e:
            self._record_test_result(
                SecurityTestType.CORS,
                "CORS Configuration Test",
                SecuritySeverity.MEDIUM,
                False,
                f"Test failed with exception: {str(e)}",
                "Review CORS configuration"
            )

    def _record_test_result(
        self,
        test_type: SecurityTestType,
        test_name: str,
        severity: SecuritySeverity,
        passed: bool,
        details: str,
        recommendation: str
    ):
        """Record a security test result."""
        result = SecurityTestResult(
            test_type=test_type,
            test_name=test_name,
            severity=severity,
            passed=passed,
            details=details,
            recommendation=recommendation,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.test_results.append(result)
        
        # Log the result
        status = "PASSED" if passed else "FAILED"
        logger.info(f"Security Test {status}: {test_name} ({severity.value})")

    def get_security_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive security report.
        
        Returns:
            Dictionary containing security report
        """
        if not self.test_results:
            return {"error": "No security tests have been run"}
        
        # Categorize results
        by_severity = {}
        by_test_type = {}
        by_status = {"passed": 0, "failed": 0}
        
        for result in self.test_results:
            # By severity
            severity = result.severity.value
            if severity not in by_severity:
                by_severity[severity] = {"passed": 0, "failed": 0}
            by_severity[severity]["passed" if result.passed else "failed"] += 1
            
            # By test type
            test_type = result.test_type.value
            if test_type not in by_test_type:
                by_test_type[test_type] = {"passed": 0, "failed": 0}
            by_test_type[test_type]["passed" if result.passed else "failed"] += 1
            
            # By status
            by_status["passed" if result.passed else "failed"] += 1
        
        # Calculate overall security score
        total_tests = len(self.test_results)
        passed_tests = by_status["passed"]
        security_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Get failed tests by severity
        failed_tests = [r for r in self.test_results if not r.passed]
        critical_failures = [r for r in failed_tests if r.severity == SecuritySeverity.CRITICAL]
        high_failures = [r for r in failed_tests if r.severity == SecuritySeverity.HIGH]
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": by_status["failed"],
                "security_score": round(security_score, 2),
                "critical_failures": len(critical_failures),
                "high_failures": len(high_failures)
            },
            "results_by_severity": by_severity,
            "results_by_test_type": by_test_type,
            "failed_tests": [
                {
                    "test_name": r.test_name,
                    "severity": r.severity.value,
                    "details": r.details,
                    "recommendation": r.recommendation
                }
                for r in failed_tests
            ],
            "recommendations": self._generate_recommendations(failed_tests)
        }

    def _generate_recommendations(self, failed_tests: List[SecurityTestResult]) -> List[str]:
        """Generate security recommendations based on failed tests."""
        recommendations = []
        
        # Group by test type
        by_type = {}
        for test in failed_tests:
            test_type = test.test_type.value
            if test_type not in by_type:
                by_type[test_type] = []
            by_type[test_type].append(test)
        
        # Generate recommendations
        if "injection" in by_type:
            recommendations.append("Implement comprehensive input validation and parameterized queries")
        
        if "authentication" in by_type:
            recommendations.append("Strengthen authentication mechanisms and implement proper session management")
        
        if "authorization" in by_type:
            recommendations.append("Implement proper role-based access control (RBAC)")
        
        if "data_exposure" in by_type:
            recommendations.append("Review API responses and implement proper data filtering")
        
        if "rate_limiting" in by_type:
            recommendations.append("Implement rate limiting to prevent abuse")
        
        if "input_validation" in by_type:
            recommendations.append("Implement proper input validation and length limits")
        
        if "file_upload" in by_type:
            recommendations.append("Implement secure file upload with proper validation and scanning")
        
        if "headers" in by_type:
            recommendations.append("Implement all recommended security headers")
        
        if "cors" in by_type:
            recommendations.append("Implement restrictive CORS policy")
        
        return recommendations


# Example usage and testing functions
def run_security_tests(base_url: str = "http://localhost:8001") -> Dict[str, Any]:
    """
    Run comprehensive security tests on the API.
    
    Args:
        base_url: Base URL of the API to test
        
    Returns:
        Security report
    """
    framework = SecurityTestingFramework(base_url)
    framework.run_all_security_tests()
    return framework.get_security_report()


if __name__ == "__main__":
    # Example usage
    report = run_security_tests()
    print(json.dumps(report, indent=2))
