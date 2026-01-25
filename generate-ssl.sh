#!/bin/bash
# ===========================================
# Generate self-signed SSL certificate for development
# For production, use Let's Encrypt or your CA
# ===========================================

SSL_DIR="./nginx/ssl"

mkdir -p "$SSL_DIR"

# Generate private key and self-signed certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout "$SSL_DIR/key.pem" \
    -out "$SSL_DIR/cert.pem" \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

echo "SSL certificates generated in $SSL_DIR"
echo ""
echo "For production, replace with real certificates from Let's Encrypt:"
echo "  certbot certonly --standalone -d yourdomain.com"
echo ""
echo "Then copy the certificates:"
echo "  cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem $SSL_DIR/cert.pem"
echo "  cp /etc/letsencrypt/live/yourdomain.com/privkey.pem $SSL_DIR/key.pem"
