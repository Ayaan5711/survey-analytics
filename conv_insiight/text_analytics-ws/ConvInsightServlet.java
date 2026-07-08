package com.tcsion.textanalyticsweb.ws.service;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.UUID;

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
        String reqId = UUID.randomUUID().toString().substring(0, 8);
        long t0 = System.currentTimeMillis();

        try {
            String contentType = request.getContentType();
            if (contentType != null && contentType.startsWith("multipart/")) {
                logger.error("[" + reqId + "] upload request received");
                String result = uploadFile(request, reqId);
                response.getWriter().write(result);
                logger.error("[" + reqId + "] upload ok in " + (System.currentTimeMillis() - t0) + "ms");
                return;
            }

            JsonNode body = mapper.readTree(request.getInputStream());
            String action = body.path("action").asText("");
            logger.error("[" + reqId + "] action=" + action + " received");

            String result = dispatch(action, body, reqId);
            response.getWriter().write(result);
            logger.error("[" + reqId + "] action=" + action + " ok in " + (System.currentTimeMillis() - t0) + "ms");

        } catch (Exception e) {
            logger.error("[" + reqId + "] request failed after " + (System.currentTimeMillis() - t0) + "ms", e);
            response.setStatus(500);
            response.getWriter().write("{\"error\":\"Failed\",\"request_id\":\"" + reqId + "\"}");
        }
    }

    // =====================================================
    // DISPATCHER
    // =====================================================
    private String dispatch(String action, JsonNode body, String reqId) throws Exception {

        switch (action) {

            case "health":
                return get("/api/health", reqId);

            case "chat":
                return post("/api/chat", body.toString(), reqId);

            case "autoInsights":
                return post("/api/auto-insights", body.toString(), reqId);

            case "dashboard":
                return post("/api/dashboard", body.toString(), reqId);

            case "report":
                return post("/api/export-report", body.toString(), reqId);

            default:
                throw new IllegalArgumentException("Unknown action: " + action);
        }
    }

    private String uploadFile(HttpServletRequest request, String reqId) throws Exception {

        HttpPost req = new HttpPost(BASE_URL + "/api/upload");
        MultipartEntityBuilder builder = MultipartEntityBuilder.create();

        // Forward every uploaded file part as "files" (supports one or many).
        boolean any = false;
        int fileCount = 0;
        for (Part part : request.getParts()) {
            String header = part.getHeader("content-disposition");
            if (header == null || !header.contains("filename=")) continue;  // skip non-file fields
            builder.addBinaryBody(
                "files",
                part.getInputStream(),
                org.apache.http.entity.ContentType.APPLICATION_OCTET_STREAM,
                getFileName(part)
            );
            any = true;
            fileCount++;
        }
        if (!any) throw new ServletException("No files found in upload");
        logger.error("[" + reqId + "] forwarding " + fileCount + " file(s) to Python layer");

        req.setEntity(builder.build());
        return execute(req, reqId);
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

    private String get(String path, String reqId) throws Exception {
        HttpGet req = new HttpGet(BASE_URL + path);
        req.setHeader("X-Request-Id", reqId);
        return execute(req, reqId);
    }

    private String post(String path, String json, String reqId) throws Exception {
        HttpPost req = new HttpPost(BASE_URL + path);
        req.setEntity(new StringEntity(json, StandardCharsets.UTF_8));
        req.setHeader("Content-Type", "application/json");
        req.setHeader("X-Request-Id", reqId);
        return execute(req, reqId);
    }

    // reqId is forwarded as X-Request-Id so the Python layer's logs can be
    // correlated with this servlet's logs for the same request.
    private String execute(HttpRequestBase req, String reqId) throws Exception {
        try (CloseableHttpResponse resp = httpClient.execute(req)) {

            int status = resp.getStatusLine().getStatusCode();
            String body = EntityUtils.toString(resp.getEntity(), StandardCharsets.UTF_8);

            if (status >= 400) {
                logger.error("[" + reqId + "] Python layer returned HTTP " + status);
                throw new RuntimeException("HTTP " + status + ": " + body);
            }

            return body != null ? body : "{}";
        }
    }
}
