document.addEventListener('DOMContentLoaded', function() {
    feather.replace();
    
    let responseTimeChart = null;
    let requestDistributionChart = null;
    
    let authToken = localStorage.getItem('erpToken');
    initializeAIModelManagement();
    
    if (!authToken) {
        showAuthModal();
    } else {
        loadDashboardData();
        loadAIModelInfo();
    }
    function showAuthModal() {
        const authModal = new bootstrap.Modal(document.getElementById('authModal'));
        authModal.show();
    }
    document.getElementById('authLoginBtn').addEventListener('click', function() {
        const username = document.getElementById('authUsername').value;
        const password = document.getElementById('authPassword').value;
        
        fetch('/api/auth/token', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({username,password})
        })
        .then(response=>response.json())
        .then(data =>{
            if (data.token){
                authToken = data.token;
                localStorage.setItem('erpToken', authToken);
                const modal = bootstrap.Modal.getInstance(document.getElementById('authModal'));
                modal.hide();
                
                loadDashboardData();
                
                loadAIModelInfo();
            } else {
                alert('Authentication failed. Please check your credentials.');
            }
        })
        .catch(error => {
            console.error('Authentication error:', error);
            alert('Error during authentication. Please try again.');
        });
    });
    
    document.getElementById('refreshBtn').addEventListener('click', function() {
        if (authToken) {
            loadDashboardData();
        } else {
            showAuthModal();
        }
    });

    function loadDashboardData() {
        updateLastUpdated();

        fetch('/api/metrics', {
            headers: {
                'Authorization': `Bearer ${authToken}`
            }
        })
        .then(response => {
            if (response.status === 401) {
                localStorage.removeItem('erpToken');
                showAuthModal();
                throw new Error('Authentication required');
            }
            return response.json();
        })
        .then(data => {
            updateDashboardMetrics(data);
            updateChartsData(data);
            updateServicesTable(data);
        })
        .catch(error => {
            console.error('Error fetching metrics:', error);
            if (error.message !== 'Authentication required') {
                alert('Error loading dashboard data. Please try again.');
            }
        });
    }
    
    function updateLastUpdated() {
        const now = new Date();
        const timeString = now.toLocaleTimeString();
        document.getElementById('lastUpdated').innerHTML = 
            `<small>Last updated: ${timeString}</small>`;
    }
    
    function initializeAIModelManagement() {
        document.getElementById('retrainBtn').addEventListener('click', function() {
            if (!authToken) {
                showAuthModal();
                return;
            }
            
            const btn = this;
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Retraining...';
            document.getElementById('retraining-status').textContent = 'Model retraining in progress...';
            
            fetch('/api/ai/retrain', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${authToken}`
                }
            })
            .then(response => {
                if (response.status === 401) {
                    localStorage.removeItem('erpToken');
                    showAuthModal();
                    throw new Error('Authentication required');
                }
                return response.json();
            })
            .then(data => {
                btn.disabled = false;
                btn.innerHTML = '<i data-feather="refresh-cw" class="feather-small"></i> Retrain Model';
                feather.replace();
                
                if (data.status === 'success') {
                    document.getElementById('retraining-status').textContent = 'Model successfully retrained! ' + 
                        'Reloading model information...';
                    
                    loadAIModelInfo();
                } else {
                    document.getElementById('retraining-status').textContent = 
                        'Model retraining failed: ' + (data.message || 'Unknown error');
                }
            })
            .catch(error => {
                console.error('Error retraining model:', error);
                btn.disabled = false;
                btn.innerHTML = '<i data-feather="refresh-cw" class="feather-small"></i> Retrain Model';
                feather.replace();
                document.getElementById('retraining-status').textContent = 
                    'Error retraining model. Please try again later.';
            });
        });

        loadAIModelInfo();
    }
    
    function loadAIModelInfo() {
        if (!authToken) {
            return;
        }
        
        fetch('/api/ai/info', {
            headers: {
                'Authorization': `Bearer ${authToken}`
            }
        })
        .then(response => {
            if (response.status === 401) {
                localStorage.removeItem('erpToken');
                showAuthModal();
                throw new Error('Authentication required');
            }
            return response.json();
        })
        .then(data => {
            const modelInfo = data.model_info || {};
            
            document.getElementById('model-version').textContent = modelInfo.version || '-';
            document.getElementById('model-accuracy').textContent = modelInfo.accuracy || '-';
            document.getElementById('training-data-points').textContent = modelInfo.training_data_points || '0';
            document.getElementById('predictions-made').textContent = modelInfo.predictions_made || '0';
        })
        .catch(error => {
            console.error('Error loading AI model info:', error);
            if (error.message !== 'Authentication required') {
                document.getElementById('retraining-status').textContent = 
                    'Error loading model information. Please try again later.';
            }
        });
    }
    
    function updateDashboardMetrics(data) {
        const apiGatewayRequests = Math.floor(Math.random() * 1000) + 100;
        const apiGatewayLoad = Math.floor(Math.random() * 70) + 10;
        
        document.getElementById('api-gateway-requests').textContent = apiGatewayRequests;
        
        const loadBar = document.getElementById('api-gateway-load');
        loadBar.style.width = `${apiGatewayLoad}%`;
        loadBar.setAttribute('aria-valuenow', apiGatewayLoad);
        
        if (apiGatewayLoad > 80) {
            document.getElementById('api-gateway-status').className = 'badge bg-danger';
            document.getElementById('api-gateway-status').textContent = 'High Load';
        } else if (apiGatewayLoad > 60) {
            document.getElementById('api-gateway-status').className = 'badge bg-warning';
            document.getElementById('api-gateway-status').textContent = 'Moderate Load';
        } else {
            document.getElementById('api-gateway-status').className = 'badge bg-success';
            document.getElementById('api-gateway-status').textContent = 'Healthy';
        }

        // if (data.adapters) {
        //     if (data.adapters.sap) {
        //         const sapAdapter = data.adapters.sap;
        //         document.getElementById('sap-adapter-requests').textContent = sapAdapter.requests;
        //         document.getElementById('sap-adapter-error-rate').textContent = `${sapAdapter.error_rate}%`;
        //         document.getElementById('sap-adapter-circuit-trips').textContent = sapAdapter.circuit_trips;

        //         const sapStatus = document.getElementById('sap-adapter-status');
        //         if (sapAdapter.status === 'healthy') {
        //             sapStatus.className = 'badge bg-success';
        //             sapStatus.textContent = 'Healthy';
        //         } else if (sapAdapter.status === 'degraded') {
        //             sapStatus.className = 'badge bg-warning';
        //             sapStatus.textContent = 'Degraded';
        //         } else {
        //             sapStatus.className = 'badge bg-danger';
        //             sapStatus.textContent = 'Unhealthy';
        //         }
        //         const sapErrorRate = document.getElementById('sap-adapter-error-rate');
        //         if (sapAdapter.error_rate > 5) {
        //             sapErrorRate.className = 'badge bg-danger';
        //         } else if (sapAdapter.error_rate > 1) {
        //             sapErrorRate.className = 'badge bg-warning';
        //         } else {
        //             sapErrorRate.className = 'badge bg-success';
        //         }
        //     }
            
        //     if (data.adapters.oracle) {
        //         const oracleAdapter = data.adapters.oracle;
        //         document.getElementById('oracle-adapter-requests').textContent = oracleAdapter.requests;
        //         document.getElementById('oracle-adapter-error-rate').textContent = `${oracleAdapter.error_rate}%`;
        //         document.getElementById('oracle-adapter-circuit-trips').textContent = oracleAdapter.circuit_trips;
                
        //         const oracleStatus = document.getElementById('oracle-adapter-status');
        //         if (oracleAdapter.status === 'healthy') {
        //             oracleStatus.className = 'badge bg-success';
        //             oracleStatus.textContent = 'Healthy';
        //         } else if (oracleAdapter.status === 'degraded') {
        //             oracleStatus.className = 'badge bg-warning';
        //             oracleStatus.textContent = 'Degraded';
        //         } else {
        //             oracleStatus.className = 'badge bg-danger';
        //             oracleStatus.textContent = 'Unhealthy';
        //         }
                
        //         const oracleErrorRate = document.getElementById('oracle-adapter-error-rate');
        //         if (oracleAdapter.error_rate > 5) {
        //             oracleErrorRate.className = 'badge bg-danger';
        //         } else if (oracleAdapter.error_rate > 1) {
        //             oracleErrorRate.className = 'badge bg-warning';
        //         } else {
        //             oracleErrorRate.className = 'badge bg-success';
        //         }
        //     }
        // }

        if (data.services && data.services.ai_predictor) {
            const aiPredictor = data.services.ai_predictor;
            document.getElementById('ai-predictor-count').textContent = aiPredictor.predictions_made || 0;
            document.getElementById('ai-predictor-version').textContent = aiPredictor.model_version || '1.0';

            const aiStatus = document.getElementById('ai-predictor-status');
            if (aiPredictor.status === 'healthy') {
                aiStatus.className = 'badge bg-success';
                aiStatus.textContent = 'Healthy';
            } else {
                aiStatus.className = 'badge bg-warning';
                aiStatus.textContent = 'Degraded';
            }
        }
        
        updateMaintenanceAlerts();
        updateRiskAlerts();
    }
    
    function updateChartsData(data) {
        const responseTimeData = {
            labels: ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00'],
            datasets: [
                {
                    label: 'SAP Adapter',
                    data: [250, 230, 240, 270, 260, 300, 280],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    tension: 0.4,
                    fill: false
                },
                {
                    label: 'Oracle Adapter',
                    data: [300, 310, 290, 320, 330, 350, 320],
                    borderColor: 'rgba(255, 159, 64, 1)',
                    tension: 0.4,
                    fill: false
                }
            ]
        };
        
        const requestDistributionData = {
            labels: ['SAP Orders', 'Oracle Orders', 'SAP Transactions', 'Oracle Transactions'],
            datasets: [{
                data: [45, 25, 20, 10],
                backgroundColor: [
                    'rgba(75, 192, 192, 0.8)',
                    'rgba(255, 159, 64, 0.8)',
                    'rgba(54, 162, 235, 0.8)',
                    'rgba(255, 99, 132, 0.8)'
                ],
                borderWidth: 1
            }]
        };

        const rtCtx = document.getElementById('responseTimeChart').getContext('2d');
        
        if (responseTimeChart) {
            responseTimeChart.data = responseTimeData;
            responseTimeChart.update();
        } else {
            responseTimeChart = new Chart(rtCtx, {
                type: 'line',
                data: responseTimeData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false,
                            title: {
                                display: true,
                                text: 'Response Time (ms)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                        },
                        title: {
                            display: false
                        }
                    }
                }
            });
        }

        const rdCtx = document.getElementById('requestDistributionChart').getContext('2d');
        
        if (requestDistributionChart) {
            requestDistributionChart.data = requestDistributionData;
            requestDistributionChart.update();
        } else {
            requestDistributionChart = new Chart(rdCtx, {
                type: 'pie',
                data: requestDistributionData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'right',
                        }
                    }
                }
            });
        }
    }
    
    function updateServicesTable(data) {
        const tableBody = document.getElementById('services-table-body');
        tableBody.innerHTML = '';

        const gatewayRow = createServiceRow('API Gateway', 'healthy', {
            requests: Math.floor(Math.random() * 1000) + 100,
            uptime: '99.9%'
        }, new Date().toISOString());
        tableBody.appendChild(gatewayRow);

        if (data.services && data.services.data_mapper) {
            const dmService = data.services.data_mapper;
            const dmRow = createServiceRow('Data Mapper', dmService.status, {
                processed: dmService.processed_count
            }, new Date().toISOString());
            tableBody.appendChild(dmRow);
        }
        
        // Add Maintenance Checker service row
        if (data.services && data.services.maintenance) {
            const mcService = data.services.maintenance;
            const mcRow = createServiceRow('Maintenance Checker', mcService.status, {
                checks: mcService.checks_performed,
                issues: mcService.issues_detected
            }, mcService.last_check_time || new Date().toISOString());
            tableBody.appendChild(mcRow);
        }

        if (data.services && data.services.ai_predictor) {
            const aiService = data.services.ai_predictor;
            const aiRow = createServiceRow('AI Predictor', aiService.status, {
                predictions: aiService.predictions_made,
                model: aiService.model_version
            }, new Date().toISOString());
            tableBody.appendChild(aiRow);
        }
        
        // Add SAP Adapter row
        // if (data.adapters && data.adapters.sap) {
        //     const sapAdapter = data.adapters.sap;
        //     const sapRow = createServiceRow('SAP Adapter', sapAdapter.status, {
        //         requests: sapAdapter.requests,
        //         errors: sapAdapter.errors,
        //         'avg response': `${sapAdapter.avg_response_time}ms`
        //     }, new Date().toISOString());
        //     tableBody.appendChild(sapRow);
        // }
        
        // // Add Oracle Adapter row
        // if (data.adapters && data.adapters.oracle) {
        //     const oracleAdapter = data.adapters.oracle;
        //     const oracleRow = createServiceRow('Oracle Adapter', oracleAdapter.status, {
        //         requests: oracleAdapter.requests,
        //         errors: oracleAdapter.errors,
        //         'avg response': `${oracleAdapter.avg_response_time}ms`
        //     }, new Date().toISOString());
        //     tableBody.appendChild(oracleRow);
        // }
    }

    function createServiceRow(name, status, metrics, lastActivity) {
        const row = document.createElement('tr');
        
        // Service name cell
        const nameCell = document.createElement('td');
        nameCell.innerText = name;
        row.appendChild(nameCell);
        
        // Status cell
        const statusCell = document.createElement('td');
        let statusBadge = document.createElement('span');
        
        if (status === 'healthy') {
            statusBadge.className = 'badge bg-success';
            statusBadge.innerText = 'Healthy';
        } else if (status === 'degraded') {
            statusBadge.className = 'badge bg-warning';
            statusBadge.innerText = 'Degraded';
        } else if (status === 'critical') {
            statusBadge.className = 'badge bg-danger';
            statusBadge.innerText = 'Critical';
        } else {
            statusBadge.className = 'badge bg-secondary';
            statusBadge.innerText = 'Unknown';
        }
        
        statusCell.appendChild(statusBadge);
        row.appendChild(statusCell);
        const metricsCell = document.createElement('td');
        const metricsList = document.createElement('small');
        
        for (const [key, value] of Object.entries(metrics)) {
            const metricItem = document.createElement('div');
            metricItem.innerHTML = `<strong>${key}:</strong> ${value}`;
            metricsList.appendChild(metricItem);
        }
        
        metricsCell.appendChild(metricsList);
        row.appendChild(metricsCell);
        
        // Last activity cell
        const activityCell = document.createElement('td');
        let activityTime;
        
        try {
            activityTime = new Date(lastActivity).toLocaleTimeString();
        } catch (e) {
            activityTime = 'Unknown';
        }
        
        activityCell.innerText = activityTime;
        row.appendChild(activityCell);
        
        // Details cell
        const detailsCell = document.createElement('td');
        const detailsButton = document.createElement('button');
        detailsButton.className = 'btn btn-sm btn-outline-secondary';
        detailsButton.innerText = 'Details';
        detailsButton.addEventListener('click', function() {
            alert(`Service details for ${name} not yet implemented.`);
        });
        
        detailsCell.appendChild(detailsButton);
        row.appendChild(detailsCell);
        
        return row;
    }
    
    function updateMaintenanceAlerts() {
        const container = document.getElementById('maintenance-alerts-container');
        
        const alerts = [
            {
                id: 1,
                type: 'missing_fields',
                message: 'Missing required fields: customer_id in SAP order data',
                severity: 'high',
                timestamp: '2023-10-21T14:32:10',
                source: 'SAP'
            },
            {
                id: 2,
                type: 'deprecated_version',
                message: 'Using deprecated Oracle API version 1.5, current is 2.1',
                severity: 'medium',
                timestamp: '2023-10-21T13:45:22',
                source: 'Oracle'
            },
            {
                id: 3,
                type: 'total_mismatch',
                message: 'Order total (450.20) doesn\'t match sum of item totals (452.80)',
                severity: 'medium',
                timestamp: '2023-10-21T12:10:05',
                source: 'SAP'
            }
        ];
        
        // Update alert count
        document.getElementById('maintenance-count').textContent = alerts.length;
        
        // Update container
        if (alerts.length === 0) {
            container.innerHTML = `
                <div class="list-group-item text-center py-4">
                    <i data-feather="check-circle" class="text-success mb-2" style="width: 32px; height: 32px;"></i>
                    <p>No maintenance alerts detected</p>
                </div>
            `;
            feather.replace();
            return;
        }
        
        // Clear container
        container.innerHTML = '';
        
        // Add alerts
        alerts.forEach(alert => {
            const alertClass = alert.severity === 'high' ? 'critical' : 
                               alert.severity === 'medium' ? 'warning' : 'info';
            
            const alertItem = document.createElement('div');
            alertItem.className = `list-group-item alert-item ${alertClass}`;
            
            // Format timestamp
            let timeAgo;
            try {
                const alertTime = new Date(alert.timestamp);
                const now = new Date();
                const diffMs = now - alertTime;
                const diffMins = Math.round(diffMs / 60000);
                
                if (diffMins < 60) {
                    timeAgo = `${diffMins} min ago`;
                } else {
                    const diffHours = Math.round(diffMins / 60);
                    timeAgo = `${diffHours} hr ago`;
                }
            } catch (e) {
                timeAgo = 'recently';
            }
            
            alertItem.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <h6 class="mb-1">${alert.source}: ${alert.type}</h6>
                    <small class="text-secondary">${timeAgo}</small>
                </div>
                <p class="mb-1">${alert.message}</p>
                <div class="d-flex justify-content-between align-items-center">
                    <small class="text-secondary">ID: ${alert.id}</small>
                    <span class="badge ${alert.severity === 'high' ? 'bg-danger' : 
                                        alert.severity === 'medium' ? 'bg-warning' : 'bg-info'}">
                        ${alert.severity}
                    </span>
                </div>
            `;
            
            container.appendChild(alertItem);
        });
    }
    
    // Update risk alerts
    function updateRiskAlerts() {
        const container = document.getElementById('risk-alerts-container');
        
        // For demonstration - in a real app this would come from API data
        const predictions = [
            {
                id: 1,
                source: 'SAP',
                risk_score: 0.78,
                risk_level: 'high',
                message: 'High risk of compatibility issues detected',
                suggestions: [
                    'Update SAP adapter to use the latest API version',
                    'Prepare for potential SAP API updates (last update was 2023-06-10)'
                ],
                predicted_at: '2023-10-21T14:30:00'
            },
            {
                id: 2,
                source: 'Oracle',
                risk_score: 0.45,
                risk_level: 'medium',
                message: 'Medium risk of compatibility issues detected',
                suggestions: [
                    'Improve field mapping to capture all required fields'
                ],
                predicted_at: '2023-10-21T13:15:00'
            }
        ];
        
        // Update risk count
        document.getElementById('risk-count').textContent = predictions.length;
        
        // Update container
        if (predictions.length === 0) {
            container.innerHTML = `
                <div class="list-group-item text-center py-4">
                    <i data-feather="shield" class="text-success mb-2" style="width: 32px; height: 32px;"></i>
                    <p>No risk predictions to display</p>
                </div>
            `;
            feather.replace();
            return;
        }
        
        container.innerHTML = '';
        
        predictions.forEach(prediction => {
            const riskClass = prediction.risk_level === 'high' ? 'critical' : 
                              prediction.risk_level === 'medium' ? 'warning' : 'info';
            
            const predictionItem = document.createElement('div');
            predictionItem.className = `list-group-item alert-item ${riskClass}`;
            
            let timeAgo;
            try {
                const predTime = new Date(prediction.predicted_at);
                const now = new Date();
                const diffMs = now - predTime;
                const diffMins = Math.round(diffMs / 60000);
                
                if (diffMins < 60) {
                    timeAgo = `${diffMins} min ago`;
                } else {
                    const diffHours = Math.round(diffMins / 60);
                    timeAgo = `${diffHours} hr ago`;
                }
            } catch (e) {
                timeAgo = 'recently';
            }
            
            const riskPercent = Math.round(prediction.risk_score*100);
            
            predictionItem.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <h6 class="mb-1">${prediction.source} Compatibility Risk</h6>
                    <small class="text-secondary">${timeAgo}</small>
                </div>
                <p class="mb-1">${prediction.message}</p>
                <div class="progress mb-2" style="height: 5px;">
                    <div class="progress-bar ${prediction.risk_level === 'high' ? 'bg-danger' : 
                                             prediction.risk_level === 'medium' ? 'bg-warning' : 'bg-info'}" 
                         role="progressbar" style="width: ${riskPercent}%;" 
                         aria-valuenow="${riskPercent}" aria-valuemin="0" aria-valuemax="100"></div>
                </div>
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <small>Risk Score</small>
                    <small>${riskPercent}%</small>
                </div>
                <div class="mt-2">
                    <small class="text-secondary">Suggestions:</small>
                    <ul class="small mb-0 ps-3">
                        ${prediction.suggestions.map(s => `<li>${s}</li>`).join('')}
                    </ul>
                </div>
            `;
            
            container.appendChild(predictionItem);
        });
    }

    setInterval(function() {
        if (authToken) {
            loadDashboardData();
        }
    }, 60000); 
});
