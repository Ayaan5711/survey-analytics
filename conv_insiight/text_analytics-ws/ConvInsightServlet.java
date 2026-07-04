package com.tcsion.textanalyticsweb.ws.service;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

import javax.servlet.*;
import javax.servlet.annotation.MultipartConfig;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.*;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.http.client.config.RequestConfig;
import org.apache.http.client.methods.*;
import org.apache.http.entity.StringEntity;
import org.apache.http.entity.mime.MultipartEntityBuilder;
import org.apache.http.impl.client.*;
import org.apache.http.util.EntityUtils;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

@WebServlet("/ConvInsightServlet")
@MultipartConfig   
public class ConvInsightServlet extends HttpServlet {

    private static final long serialVersionUID = 1L;
    private static final String BASE_URL = "http://127.0.0.1:8001";
    private static final int TIMEOUT_MS = 600_000;

    private final Log logger = LogFactory.getLog(ConvInsightServlet.class);
    private final ObjectMapper mapper = new ObjectMapper();

    private CloseableHttpClient httpClient;

    @Override
    public void init() throws ServletException {
        RequestConfig config = RequestConfig.custom()
                .setConnectTimeout(TIMEOUT_MS)
                .setSocketTimeout(TIMEOUT_MS)
                .setConnectionRequestTimeout(TIMEOUT_MS)
                .build();

        httpClient = HttpClients.custom()
                .setDefaultRequestConfig(config)
                .build();
    }

    @Override
    public void destroy() {
        try {
            if (httpClient != null) httpClient.close();
        } catch (IOException e) {
            logger.error("Error closing HTTP client", e);
        }
    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        response.setContentType("application/json;charset=UTF-8");

        try {
            String contentType = request.getContentType();
            if (contentType != null && contentType.startsWith("multipart/")) {
                String result = uploadFile(request);
                response.getWriter().write(result);
                return;
            }

            JsonNode body = mapper.readTree(request.getInputStream());
            String action = body.path("action").asText("");

            String result = dispatch(action, body);
            response.getWriter().write(result);

        } catch (Exception e) {
            logger.error("Error processing request", e);
            response.setStatus(500);
            response.getWriter().write("{\"error\":\"Failed\"}");
        }
    }

    // =====================================================
    // DISPATCHER
    // =====================================================
    private String dispatch(String action, JsonNode body) throws Exception {

        switch (action) {

            case "health":
                return get("/api/health");

            case "chat":
                return post("/api/chat", body.toString());

            case "autoInsights":
                return post("/api/auto-insights", body.toString());

            default:
                throw new IllegalArgumentException("Unknown action: " + action);
        }
    }

    private String uploadFile(HttpServletRequest request) throws Exception {

        Part filePart = request.getPart("file");

        HttpPost req = new HttpPost(BASE_URL + "/api/upload");

        MultipartEntityBuilder builder = MultipartEntityBuilder.create();

        builder.addBinaryBody(
            "file",
            filePart.getInputStream(),
            org.apache.http.entity.ContentType.APPLICATION_OCTET_STREAM,
            getFileName(filePart) 
        );

        req.setEntity(builder.build());

        return execute(req);
    }
    
    private String getFileName(Part part) {
        String header = part.getHeader("content-disposition");
        if (header == null) return "file.xlsx";

        for (String s : header.split(";")) {
            if (s.trim().startsWith("filename")) {
                return s.substring(s.indexOf("=") + 1).replace("\"", "");
            }
        }
        return "file.xlsx";
    }



    // =====================================================
    // HTTP HELPERS
    // =====================================================

    private String get(String path) throws Exception {
        return execute(new HttpGet(BASE_URL + path));
    }

    private String post(String path, String json) throws Exception {
        HttpPost req = new HttpPost(BASE_URL + path);
        req.setEntity(new StringEntity(json, StandardCharsets.UTF_8));
        req.setHeader("Content-Type", "application/json");
        return execute(req);
    }

    private String execute(HttpRequestBase req) throws Exception {
        try (CloseableHttpResponse resp = httpClient.execute(req)) {

            int status = resp.getStatusLine().getStatusCode();
            String body = EntityUtils.toString(resp.getEntity(), StandardCharsets.UTF_8);

            if (status >= 400) {
                throw new RuntimeException("HTTP " + status + ": " + body);
            }

            return body != null ? body : "{}";
        }
    }
}
