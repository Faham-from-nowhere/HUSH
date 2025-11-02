# HUSH: Privacy & Ethics Appendix

Our privacy-first architecture is a core feature, not an afterthought. We've designed HUSH to be trustworthy by default.

## 1. On-Device Processing
All sensitive user data—including journal text, audio clips, and raw typing patterns—is **processed locally on the user's device**. This raw, sensitive data **never** leaves the device and is **never** sent to our servers.

## 2. Federated Learning (FL)
Instead of uploading user data for training, our models learn on the device. Only anonymous, aggregated *model updates* (e.g., "text was 70% important for this user") are sent to our backend.
* **Our Simulation:** Our backend's `/v1/submit-update` endpoint simulates this by accepting *only* these feature attributions, not the raw data itself. We use **Federated Averaging (FedAvg)** to combine these updates into a smarter global model.

## 3. Differential Privacy (DP)
To protect against "model-inversion" attacks, we add statistical "noise" to every update *before* it's aggregated.
* **Our Simulation:** We use the **Laplace Mechanism** (`np.random.laplace`) to add this privacy-preserving noise. This makes it mathematically impossible to reverse-engineer any single individual's contribution from the global model, even if our backend was compromised.

## 4. Data Minimization & Transparency
* **Backend:** The only data we store is the *anonymized, aggregated* global model weights over time. This is what powers the admin dashboard. We **do not** store any user-identifiable information.
* **Frontend:** The user has full control to see what data is used and to opt-out of any data modality (text, audio, typing) at any time.