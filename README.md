# 3D Penrose Tiling

A web application using Rust Lambda, Three.js frontend, and AWS infrastructure.

## Local Development

### Prerequisites
- Rust and cargo-lambda (`cargo install cargo-lambda`)
- Node.js and npm
- AWS CDK CLI (`npm install -g aws-cdk`)
- AWS CLI configured with your credentials

### Backend (Rust Lambda)
From `/rust` directory:
1. `cargo lambda build` - Build the Lambda function
2. `cargo lambda watch` - Start local Lambda server
3. Test at http://localhost:9000/lambda-url/dualgrid/

### Frontend (Three.js)
From `/web` directory:
1. `npm install` - Install dependencies
2. `npm run dev` - Start Vite dev server
3. Visit http://localhost:5173

The dev server automatically proxies API requests to the local Lambda.

## Deployment

1. Build Lambda (from `/rust`):
   `cargo lambda build --release --arm64`

2. Build Frontend (from `/web`):
   `npm run build`

3. Deploy Infrastructure (from `/cdk`):
   ```
   npm install
   cdk deploy
   ```

The deployment will:
- Create/update the Lambda function
- Deploy API Gateway and CloudFront
- Upload frontend to S3
- Output the CloudFront URL for your application

## Project Structure
```
project/
├── rust/           # Rust Lambda backend
├── web/            # Three.js frontend
└── cdk/            # AWS infrastructure
```

## Development Notes
- Frontend uses Vite for bundling and development
- Lambda uses custom Rust runtime
- Infrastructure managed with AWS CDK
- CloudFront distribution serves both static files and API

## Environment Variables
Frontend:
- `VITE_API_URL`: API endpoint (automatically handled in development)

## Useful Commands
- `cdk diff` - Show infrastructure changes
- `cdk destroy` - Remove AWS resources
- `cargo lambda deploy dualgrid` - Update Lambda code only

## Contributing
[Add contribution guidelines]

## License
[Add license information]
