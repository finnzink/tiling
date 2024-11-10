backend:

from inside the rust dir, run `cargo lambda build` and `cargo lambda watch` to get the lambda running locally, which you can then invoke via `http://localhost:9000/lambda-url/dualgrid/`

from inside the web dir, run `npm run dev`, this will run the frontend with vite