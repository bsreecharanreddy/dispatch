pub mod dispatch_v1 {
    tonic::include_proto!("dispatch.v1");
}

#[cfg(test)]
mod tests {
    use prost::Message;

    use super::dispatch_v1::GenerateRequest;

    #[test]
    fn generate_request_round_trips_through_prost_encoding() {
        let original = GenerateRequest {
            request_id: "r1".to_string(),
            prompt: "hello".to_string(),
            max_new_tokens: 16,
        };

        let mut buf = Vec::new();
        original.encode(&mut buf).expect("encode");
        let decoded = GenerateRequest::decode(buf.as_slice()).expect("decode");

        assert_eq!(decoded, original);
    }
}
