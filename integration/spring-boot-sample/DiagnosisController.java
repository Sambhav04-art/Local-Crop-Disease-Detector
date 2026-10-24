package com.example.agri.diagnosis;

import com.example.agri.cropdx.*;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.springframework.web.server.ResponseStatusException;

interface DiagnosisRepository extends MongoRepository<DiagnosisRecord, String> {}

@RestController
@RequestMapping("/api/diagnoses")
public class DiagnosisController {

    private static final Set<String> OK_TYPES = Set.of("image/jpeg", "image/png", "image/webp");
    private final CropDiseaseClient client;
    private final DiagnosisRepository repo;

    public DiagnosisController(CropDiseaseClient client, DiagnosisRepository repo) {
        this.client = client; this.repo = repo;
    }

    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public DiagnosisRecord create(@RequestParam("image") MultipartFile image) throws IOException {
        if (image.isEmpty() || !OK_TYPES.contains(image.getContentType()))
            throw new ResponseStatusException(HttpStatus.UNSUPPORTED_MEDIA_TYPE, "Send a JPEG/PNG/WebP image");
        DiagnosisRecord rec = new DiagnosisRecord();
        rec.result = client.predict(image, 3);
        // rec.imageRef = gridFs.store(image);   // TODO store the image
        return repo.save(rec);
    }

    @ExceptionHandler(CropDxException.class)
    ResponseEntity<Map<String, String>> onCropDx(CropDxException e) {
        // 400 bad image -> 422 for your clients; 503/timeouts -> 503 "try again later"
        HttpStatus s = e.getHttpStatus() == 400 ? HttpStatus.UNPROCESSABLE_ENTITY
                     : e.getHttpStatus() == 413 ? HttpStatus.PAYLOAD_TOO_LARGE : HttpStatus.SERVICE_UNAVAILABLE;
        return ResponseEntity.status(s).body(Map.of("code", e.getCode(), "message", e.getMessage()));
    }
}
