package com.example.agri.cropdx;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClient;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.time.Duration;
import java.util.UUID;

/** Spring Boot 3.2+ (RestClient). Calls the Python prediction service. */
@Service
public class CropDiseaseClient {

    private final RestClient rest;
    private final ObjectMapper mapper;

    public CropDiseaseClient(@Value("${cropdx.base-url}") String baseUrl,
                             @Value("${cropdx.api-key:}") String apiKey,
                             ObjectMapper mapper) {
        this.mapper = mapper;
        SimpleClientHttpRequestFactory f = new SimpleClientHttpRequestFactory();
        f.setConnectTimeout(Duration.ofSeconds(3));
        f.setReadTimeout(Duration.ofSeconds(15));
        RestClient.Builder b = RestClient.builder().baseUrl(baseUrl).requestFactory(f);
        if (!apiKey.isBlank()) b.defaultHeader("X-API-Key", apiKey);
        this.rest = b.build();
    }

    public PredictionResponse predict(MultipartFile image, int topK) throws IOException {
        ByteArrayResource part = new ByteArrayResource(image.getBytes()) {
            @Override public String getFilename() {
                return image.getOriginalFilename() == null ? "leaf.jpg" : image.getOriginalFilename();
            }
        };
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", part);                      // field name MUST be "file"

        return rest.post()
                .uri("/v1/predict?topK={k}", topK)
                .header("X-Request-Id", UUID.randomUUID().toString())
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(body)
                .retrieve()
                .onStatus(s -> s.isError(), (req, res) -> {
                    String code = "UNKNOWN", msg = "Prediction service error";
                    try {
                        JsonNode err = mapper.readTree(res.getBody()).path("error");
                        code = err.path("code").asText(code);
                        msg = err.path("message").asText(msg);
                    } catch (Exception ignored) { }
                    throw new CropDxException(res.getStatusCode().value(), code, msg);
                })
                .body(PredictionResponse.class);
    }
}
