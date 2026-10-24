package com.example.agri.diagnosis;

import com.example.agri.cropdx.PredictionResponse;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.Instant;

/** Store the WHOLE model answer + model version for audit/retraining. Keep the image in GridFS/S3, not in this doc. */
@Document("diagnoses")
public class DiagnosisRecord {
    @Id public String id;
    public String userId;
    public String imageRef;                 // GridFS id / object-storage key
    public Instant createdAt = Instant.now();
    public PredictionResponse result;       // embeds status, prediction, candidates, modelVersion, requestId...
    public Boolean userConfirmedCorrect;    // optional feedback loop -> future retraining data
}
