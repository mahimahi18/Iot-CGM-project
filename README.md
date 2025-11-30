# LIVE-MART- Objective

Our aim for the LIVE-MART project was to build a simple, reliable, and scalable delivery system. To achieve this, we went ahead with Firebase + React as our tech stack. 

# Stack
On the backend, we used Firebase Firestore as our NoSQL database for its real-time syncing and easier integration with other Firebase services. User authentication and OTP-based mobile authentication is handled through Firebase Authentication (OAuth 2.0), where we also integrated Google social login as an option for user-friendly social logins.
The frontend is built using React, allowing us to develop reusable UI components and efficiently manage dynamic state across the application. For location-based features such as store locations, order distance calculations, and delivery mapping, we integrated the Google Maps JavaScript API.
In order to support automated email flows for events like order confirmations, password recovery, and verification, we used Twilio’s SendGrid (SMTP), which integrates with Firebase Cloud Functions for server-side triggers. The application follows a Hybrid Cloud Architecture.

