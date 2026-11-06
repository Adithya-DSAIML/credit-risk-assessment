(() => {
  const form = document.getElementById('loan-form');
  const submitButton = document.getElementById('submit-button');
  const formError = document.getElementById('form-error');
  const resultEmpty = document.getElementById('result-empty');
  const resultContent = document.getElementById('result-content');
  const detailsToggle = document.getElementById('details-toggle');
  const submittedDetails = document.getElementById('submitted-details');

  // Same-origin by default: FastAPI serves both this page and /predict.
  // For a separately hosted API, set window.CREDIT_RISK_API_URL before app.js loads.
  const API_BASE_URL = (window.CREDIT_RISK_API_URL || window.location.origin).replace(/\/$/, '');

  document.getElementById('year').textContent = new Date().getFullYear();

  const formatPercent = (value) => `${(Number(value) * 100).toFixed(2)}%`;
  const displayValue = (value) => value === null || value === undefined || value === '' ? 'Not provided' : String(value);

  function setLoading(isLoading) {
    submitButton.classList.toggle('is-loading', isLoading);
    submitButton.disabled = isLoading;
    submitButton.querySelector('.button-label').textContent = isLoading ? 'Assessing application' : 'Assess application';
  }

  function showError(message) {
    formError.textContent = message;
    formError.hidden = false;
  }

  function clearError() {
    formError.textContent = '';
    formError.hidden = true;
  }

  function buildPayload() {
    const values = new FormData(form);
    const employmentLength = values.get('person_emp_length');
    return {
      person_age: Number(values.get('person_age')),
      person_income: Number(values.get('person_income')),
      person_home_ownership: values.get('person_home_ownership'),
      person_emp_length: employmentLength === '' ? null : Number(employmentLength),
      loan_intent: values.get('loan_intent'),
      loan_grade: values.get('loan_grade'),
      loan_amnt: Number(values.get('loan_amnt')),
      loan_int_rate: Number(values.get('loan_int_rate')),
      loan_percent_income: Number(values.get('loan_percent_income')),
      cb_person_default_on_file: values.get('cb_person_default_on_file'),
      cb_person_cred_hist_length: Number(values.get('cb_person_cred_hist_length'))
    };
  }

  function renderDetails(features) {
    const labels = {
      person_age: 'Applicant age',
      person_income: 'Annual income',
      person_home_ownership: 'Home ownership',
      person_emp_length: 'Employment length (years)',
      loan_intent: 'Loan purpose',
      loan_grade: 'Loan grade',
      loan_amnt: 'Loan amount',
      loan_int_rate: 'Interest rate',
      loan_percent_income: 'Loan / income ratio',
      cb_person_default_on_file: 'Previous default on file',
      cb_person_cred_hist_length: 'Credit history (years)'
    };
    submittedDetails.replaceChildren();
    const record = Array.isArray(features) ? (features[0] || {}) : (features || {});
    Object.entries(labels).forEach(([key, label]) => {
      const row = document.createElement('div');
      row.className = 'detail-row';
      const name = document.createElement('span');
      const value = document.createElement('span');
      name.textContent = label;
      value.textContent = displayValue(record[key]);
      row.append(name, value);
      submittedDetails.append(row);
    });
  }

  function showResult(result) {
    const probability = Number(result.default_probability);
    const threshold = Number(result.classification_threshold);
    if (!Number.isFinite(probability) || probability < 0 || probability > 1 || !Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
      throw new Error('The API returned an invalid probability or threshold. Check the response schema and model output.');
    }

    const isDefault = Number(result.prediction) === 1;
    const label = isDefault ? 'Default' : 'Non-default';

    resultEmpty.hidden = true;
    resultContent.hidden = false;

    const decisionTitle = document.getElementById('prediction-label');
    const decisionIcon = document.getElementById('decision-icon');
    decisionTitle.textContent = result.prediction_label || label;
    decisionTitle.classList.toggle('non-default', !isDefault);
    decisionIcon.classList.toggle('non-default', !isDefault);
    decisionIcon.textContent = isDefault ? '!' : '✓';
    document.getElementById('decision-caption').textContent = isDefault
      ? 'The model classified this application as default.'
      : 'The model classified this application as non-default.';

    const percentage = probability * 100;
    document.getElementById('probability-value').textContent = percentage.toFixed(2);
    document.getElementById('threshold-value').textContent = formatPercent(threshold);
    document.getElementById('class-value').textContent = String(result.prediction);
    document.getElementById('threshold-track-label').textContent = `Threshold ${formatPercent(threshold)}`;

    const fill = document.getElementById('probability-fill');
    fill.classList.toggle('non-default', !isDefault);
    fill.style.width = '0%';
    requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = `${percentage}%`; }));
    document.getElementById('threshold-marker').style.left = `${threshold * 100}%`;

    const explanation = document.querySelector('#result-explainer p');
    explanation.textContent = isDefault
      ? `The estimated default probability (${formatPercent(probability)}) is at or above the classification threshold (${formatPercent(threshold)}).`
      : `The estimated default probability (${formatPercent(probability)}) is below the classification threshold (${formatPercent(threshold)}).`;

    renderDetails(result.received_features);
    submittedDetails.hidden = true;
    detailsToggle.setAttribute('aria-expanded', 'false');
    detailsToggle.querySelector('span').textContent = '＋';
    resultContent.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function extractApiError(payload, status) {
    if (payload && Array.isArray(payload.detail)) {
      return payload.detail.map((item) => {
        const field = Array.isArray(item.loc) ? item.loc.slice(1).join('.') : 'Input';
        return `${field}: ${item.msg || 'Invalid value'}`;
      }).join(' · ');
    }
    if (payload && typeof payload.detail === 'string') return payload.detail;
    if (payload && typeof payload.message === 'string') return payload.message;
    return `The API request failed (HTTP ${status}). Please check that the service is running and try again.`;
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    clearError();
    if (!form.reportValidity()) return;

    const payload = buildPayload();
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify(payload)
      });

      let result;
      try {
        result = await response.json();
      } catch {
        throw new Error(`The API returned a response that was not valid JSON (HTTP ${response.status}).`);
      }

      if (!response.ok) throw new Error(extractApiError(result, response.status));
      showResult(result);
    } catch (error) {
      showError(error instanceof TypeError
        ? 'Could not reach the prediction API. Confirm that the FastAPI service is running and that the frontend is being opened from the same service.'
        : error.message || 'Something went wrong while assessing this application.');
    } finally {
      setLoading(false);
    }
  });

  detailsToggle.addEventListener('click', () => {
    const isExpanded = detailsToggle.getAttribute('aria-expanded') === 'true';
    detailsToggle.setAttribute('aria-expanded', String(!isExpanded));
    submittedDetails.hidden = isExpanded;
    detailsToggle.querySelector('span').textContent = isExpanded ? '＋' : '−';
  });
})();
