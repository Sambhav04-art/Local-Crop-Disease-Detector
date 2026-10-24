package com.example.agri.cropdx;

public class CropDxException extends RuntimeException {
    private final int httpStatus;
    private final String code;
    public CropDxException(int httpStatus, String code, String message) {
        super(message); this.httpStatus = httpStatus; this.code = code;
    }
    public int getHttpStatus() { return httpStatus; }
    public String getCode() { return code; }
}
