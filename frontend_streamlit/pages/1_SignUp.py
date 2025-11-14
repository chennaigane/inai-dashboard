import streamlit as st
import smtplib
from email.message import EmailMessage

# --- Logo top-left ---
logo_file = "Inal-Logo.png"  # Update with your logo file name if needed!
st.markdown(f"""
    <div style="position: fixed; left: 25px; top: 18px; z-index:99;">
        <img src="{logo_file}" width="68"/>
    </div>
    """, unsafe_allow_html=True)

st.header("Sign Up for InaI")

with st.form("signup_form"):
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    password = st.text_input("Create Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    submitted = st.form_submit_button("Sign Up")

    if submitted:
        if not name or not email or not password or not confirm_password:
            st.error("Please fill all fields.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        else:
            # --- Email to Admin for Approval ---
            admin_email = "info1@icondf.com"
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            sender_email = "your_app_email@gmail.com"   # Replace with your sender email
            sender_password = "your_password"           # Replace with your app password

            msg = EmailMessage()
            msg['Subject'] = "InaI Sign Up Request"
            msg['From'] = sender_email
            msg['To'] = admin_email
            msg.set_content(
                f"New user sign-up request:\n"
                f"Name: {name}\n"
                f"Email: {email}\n"
                "Please approve access for this user."
            )
            try:
                # Uncomment for live use with valid SMTP credentials
                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
                server.quit()

                
                st.success("Sign-up request sent! You'll get access after admin approval via email.")
            except Exception as e:
                st.error(f"Could not send sign-up email: {e}")

st.info("After submitting, your request will be sent to the admin for approval. You will be notified once access is granted.")
