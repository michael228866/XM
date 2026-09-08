package local.xm;

import com.dukascopy.api.Filter;
import com.dukascopy.api.IAccount;
import com.dukascopy.api.IBar;
import com.dukascopy.api.IContext;
import com.dukascopy.api.IHistory;
import com.dukascopy.api.IMessage;
import com.dukascopy.api.IStrategy;
import com.dukascopy.api.ITick;
import com.dukascopy.api.Instrument;
import com.dukascopy.api.JFException;
import com.dukascopy.api.LoadingDataListener;
import com.dukascopy.api.LoadingProgressListener;
import com.dukascopy.api.OfferSide;
import com.dukascopy.api.Period;
import com.dukascopy.api.RequiresFullAccess;
import com.dukascopy.api.system.ClientFactory;
import com.dukascopy.api.system.IClient;
import com.dukascopy.api.system.ISystemListener;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.zip.GZIPOutputStream;

public final class JForexFinalCollector {
    private static final long HOUR_MS = 3_600_000L;
    private static final long MINUTE_MS = 60_000L;
    private static final String API_VERSION = "2.13.99";
    private static final String CLIENT_VERSION = "3.6.51";

    private JForexFinalCollector() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            throw new IllegalArgumentException("usage: <request.tsv> <output-directory>");
        }
        String username = requiredEnvironment("DUKASCOPY_USERNAME");
        String password = requiredEnvironment("DUKASCOPY_PASSWORD");
        String jnlp = System.getenv().getOrDefault(
                "DUKASCOPY_JNLP", "https://platform.dukascopy.com/demo_3/jforex_3.jnlp");
        Path requestPath = Paths.get(args[0]).toAbsolutePath().normalize();
        Path outputPath = Paths.get(args[1]).toAbsolutePath().normalize();
        Files.createDirectories(outputPath);
        List<Request> requests = readRequests(requestPath);
        if (requests.size() != 784) {
            throw new IllegalStateException("expected 784 frozen requests, found " + requests.size());
        }

        IClient client = ClientFactory.getDefaultInstance();
        CountDownLatch stopped = new CountDownLatch(1);
        AtomicReference<Throwable> failure = new AtomicReference<Throwable>();
        client.setSystemListener(new ISystemListener() {
            public void onStart(long processId) {
            }

            public void onStop(long processId) {
                stopped.countDown();
            }

            public void onConnect() {
            }

            public void onDisconnect() {
            }
        });
        try {
            client.connect(jnlp, username, password);
            for (int i = 0; i < 45 && !client.isConnected(); i++) {
                Thread.sleep(1_000L);
            }
            if (!client.isConnected()) {
                throw new IllegalStateException("JForex connection timeout");
            }
            Set<Instrument> instruments = new HashSet<Instrument>(Arrays.asList(
                    Instrument.EURUSD, Instrument.GBPUSD, Instrument.USDJPY));
            client.setSubscribedInstruments(instruments);
            client.startStrategy(new CollectorStrategy(requests, outputPath, failure));
            if (!stopped.await(6L, TimeUnit.HOURS)) {
                throw new IllegalStateException("JForex collection timeout after 6 hours");
            }
            if (failure.get() != null) {
                throw new IllegalStateException("collector failed", failure.get());
            }
            System.out.println("JFOREX_FINAL_COLLECTION_PASS");
        } finally {
            client.disconnect();
        }
    }

    private static String requiredEnvironment(String name) {
        String value = System.getenv(name);
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalStateException(name + " is missing");
        }
        return value;
    }

    private static List<Request> readRequests(Path path) throws IOException {
        List<Request> result = new ArrayList<Request>();
        try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String header = reader.readLine();
            if (!"request_id\tpair\tfrom_ms\tto_ms\tkind".equals(header)) {
                throw new IOException("unexpected request manifest header");
            }
            String line;
            while ((line = reader.readLine()) != null) {
                String[] fields = line.split("\\t", -1);
                if (fields.length != 5) {
                    throw new IOException("invalid request row: " + line);
                }
                result.add(new Request(
                        fields[0], fields[1], Long.parseLong(fields[2]),
                        Long.parseLong(fields[3]), fields[4]));
            }
        }
        return result;
    }

    @RequiresFullAccess
    private static final class CollectorStrategy implements IStrategy {
        private final List<Request> requests;
        private final Path output;
        private final AtomicReference<Throwable> failure;
        private IContext context;

        CollectorStrategy(List<Request> requests, Path output, AtomicReference<Throwable> failure) {
            this.requests = requests;
            this.output = output;
            this.failure = failure;
        }

        public void onStart(IContext strategyContext) throws JFException {
            context = strategyContext;
            context.setSubscribedInstruments(new HashSet<Instrument>(Arrays.asList(
                    Instrument.EURUSD, Instrument.GBPUSD, Instrument.USDJPY)), true);
            Thread worker = new Thread(new Runnable() {
                public void run() {
                    try {
                        collect(context.getHistory());
                    } catch (Throwable error) {
                        failure.set(error);
                        error.printStackTrace(System.err);
                    } finally {
                        context.stop();
                    }
                }
            }, "jforex-final-collector");
            worker.setDaemon(true);
            worker.start();
        }

        private void collect(IHistory history) throws Exception {
            Path tickPath = output.resolve("jforex_ticks.tsv.gz");
            Path barPath = output.resolve("jforex_bars.tsv.gz");
            Path auditPath = output.resolve("jforex_request_audit.tsv");
            try (BufferedWriter ticks = gzipWriter(tickPath);
                 BufferedWriter bars = gzipWriter(barPath);
                 BufferedWriter audit = Files.newBufferedWriter(auditPath, StandardCharsets.UTF_8)) {
                ticks.write("request_id\tpair\tmechanism\tsequence\ttime_ms\tbid\task\tbid_volume\task_volume\n");
                bars.write("request_id\tpair\tbar_time_ms\topen\thigh\tlow\tclose\tvolume\tflat\n");
                audit.write("request_id\tpair\tfrom_ms\tto_ms\tkind\tmechanism\tattempt\tstatus\tall_data_loaded\tcount\tfirst_time_ms\tfirst_bid\tlast_time_ms\tlast_bid\texact_from_ticks\texact_to_ticks\traw_sha256\tacquired_utc\terror\tapi_version\tclient_version\tjava_version\n");
                for (int index = 0; index < requests.size(); index++) {
                    Request request = requests.get(index);
                    Instrument instrument = instrument(request.pair);
                    collectBlockingTicks(history, instrument, request, ticks, audit);
                    collectAsyncTicks(history, instrument, request, ticks, audit);
                    collectBars(history, instrument, request, bars, audit);
                    if ((index + 1) % 25 == 0 || index + 1 == requests.size()) {
                        ticks.flush();
                        bars.flush();
                        audit.flush();
                        System.out.println("JFOREX_PROGRESS=" + (index + 1) + "/" + requests.size());
                    }
                }
            }
        }

        private void collectBlockingTicks(
                IHistory history, Instrument instrument, Request request,
                BufferedWriter tickWriter, BufferedWriter auditWriter) throws IOException {
            List<TickRecord> records = new ArrayList<TickRecord>();
            String status = "success";
            String error = "";
            try {
                for (ITick tick : history.getTicks(instrument, request.fromMs, request.toMs)) {
                    records.add(new TickRecord(tick.getTime(), tick.getBid(), tick.getAsk(),
                            tick.getBidVolume(), tick.getAskVolume()));
                }
            } catch (Throwable thrown) {
                status = "error";
                error = cleanError(thrown);
            }
            writeTicks(request, "getTicks", records, tickWriter);
            writeAudit(request, "getTicks", 1, status, "not_applicable", records, error, auditWriter);
        }

        private void collectAsyncTicks(
                IHistory history, Instrument instrument, Request request,
                BufferedWriter tickWriter, BufferedWriter auditWriter) throws IOException {
            List<TickRecord> records = Collections.synchronizedList(new ArrayList<TickRecord>());
            CountDownLatch loaded = new CountDownLatch(1);
            AtomicReference<Boolean> allLoaded = new AtomicReference<Boolean>();
            AtomicReference<String> asyncError = new AtomicReference<String>("");
            String status = "success";
            try {
                history.readTicks(instrument, request.fromMs, request.toMs,
                        new LoadingDataListener() {
                            public void newTick(Instrument ignored, long time, double ask, double bid,
                                                double askVol, double bidVol) {
                                records.add(new TickRecord(time, bid, ask, bidVol, askVol));
                            }

                            public void newBar(Instrument ignored, Period period, OfferSide side,
                                               long time, double open, double close,
                                               double low, double high, double volume) {
                            }
                        },
                        new LoadingProgressListener() {
                            public void dataLoaded(long start, long end, long current, String information) {
                            }

                            public void loadingFinished(boolean complete, long start, long end, long current) {
                                allLoaded.set(Boolean.valueOf(complete));
                                loaded.countDown();
                            }

                            public boolean stopJob() {
                                return false;
                            }
                        });
                if (!loaded.await(5L, TimeUnit.MINUTES)) {
                    status = "error";
                    asyncError.set("TimeoutException: readTicks exceeded 5 minutes");
                } else if (!Boolean.TRUE.equals(allLoaded.get())) {
                    status = "error";
                    asyncError.set("JFHistoryError: loadingFinished(allDataLoaded=false)");
                }
            } catch (Throwable thrown) {
                status = "error";
                asyncError.set(cleanError(thrown));
            }
            List<TickRecord> snapshot;
            synchronized (records) {
                snapshot = new ArrayList<TickRecord>(records);
            }
            writeTicks(request, "readTicks", snapshot, tickWriter);
            writeAudit(request, "readTicks", 1, status,
                    allLoaded.get() == null ? "unknown" : allLoaded.get().toString(),
                    snapshot, asyncError.get(), auditWriter);
        }

        private void collectBars(
                IHistory history, Instrument instrument, Request request,
                BufferedWriter barWriter, BufferedWriter auditWriter) throws IOException {
            List<IBar> result = new ArrayList<IBar>();
            String status = "success";
            String error = "";
            long barFrom = request.fromMs;
            long barTo = request.toMs - MINUTE_MS;
            try {
                result.addAll(history.getBars(instrument, Period.ONE_MIN, OfferSide.BID,
                        Filter.NO_FILTER, barFrom, barTo));
            } catch (Throwable thrown) {
                status = "error";
                error = cleanError(thrown);
            }
            List<TickRecord> digestRecords = new ArrayList<TickRecord>();
            for (int i = 0; i < result.size(); i++) {
                IBar bar = result.get(i);
                barWriter.write(request.id + "\t" + request.pair + "\t" + bar.getTime()
                        + "\t" + bar.getOpen() + "\t" + bar.getHigh() + "\t" + bar.getLow()
                        + "\t" + bar.getClose() + "\t" + bar.getVolume() + "\t"
                        + Boolean.toString(bar.getOpen() == bar.getHigh()
                        && bar.getOpen() == bar.getLow() && bar.getOpen() == bar.getClose()) + "\n");
                digestRecords.add(new TickRecord(bar.getTime(), bar.getOpen(), bar.getClose(),
                        bar.getLow(), bar.getHigh()));
            }
            writeAudit(request, "getBars", 1, status, "not_applicable",
                    digestRecords, error, auditWriter);
        }

        private static void writeTicks(
                Request request, String mechanism, List<TickRecord> records,
                BufferedWriter writer) throws IOException {
            for (int index = 0; index < records.size(); index++) {
                TickRecord tick = records.get(index);
                writer.write(request.id + "\t" + request.pair + "\t" + mechanism + "\t"
                        + index + "\t" + tick.time + "\t" + tick.bid + "\t" + tick.ask
                        + "\t" + tick.bidVolume + "\t" + tick.askVolume + "\n");
            }
        }

        private static void writeAudit(
                Request request, String mechanism, int attempt, String status,
                String allDataLoaded, List<TickRecord> records, String error,
                BufferedWriter writer) throws IOException {
            TickRecord first = records.isEmpty() ? null : records.get(0);
            TickRecord last = records.isEmpty() ? null : records.get(records.size() - 1);
            long exactFrom = 0L;
            long exactTo = 0L;
            for (TickRecord tick : records) {
                if (tick.time == request.fromMs) {
                    exactFrom++;
                }
                if (tick.time == request.toMs) {
                    exactTo++;
                }
            }
            writer.write(request.id + "\t" + request.pair + "\t" + request.fromMs + "\t"
                    + request.toMs + "\t" + request.kind + "\t" + mechanism + "\t" + attempt
                    + "\t" + status + "\t" + allDataLoaded + "\t" + records.size() + "\t"
                    + value(first, true) + "\t" + price(first) + "\t" + value(last, true)
                    + "\t" + price(last) + "\t" + exactFrom + "\t" + exactTo + "\t"
                    + sha256(records) + "\t" + Instant.now().toString() + "\t" + clean(error)
                    + "\t" + API_VERSION + "\t" + CLIENT_VERSION + "\t"
                    + clean(System.getProperty("java.runtime.version")) + "\n");
        }

        private static String value(TickRecord record, boolean time) {
            return record == null ? "" : Long.toString(record.time);
        }

        private static String price(TickRecord record) {
            return record == null ? "" : Double.toString(record.bid);
        }

        private static String sha256(List<TickRecord> records) {
            try {
                ByteArrayOutputStream bytes = new ByteArrayOutputStream();
                DataOutputStream data = new DataOutputStream(bytes);
                for (TickRecord record : records) {
                    data.writeLong(record.time);
                    data.writeDouble(record.bid);
                    data.writeDouble(record.ask);
                    data.writeDouble(record.bidVolume);
                    data.writeDouble(record.askVolume);
                }
                data.flush();
                MessageDigest digest = MessageDigest.getInstance("SHA-256");
                byte[] value = digest.digest(bytes.toByteArray());
                StringBuilder result = new StringBuilder();
                for (byte item : value) {
                    result.append(String.format(Locale.ROOT, "%02x", item & 0xff));
                }
                return result.toString();
            } catch (Exception error) {
                throw new IllegalStateException(error);
            }
        }

        private static String cleanError(Throwable error) {
            return clean(error.getClass().getName() + ": " + String.valueOf(error.getMessage()));
        }

        private static String clean(String value) {
            return value == null ? "" : value.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ');
        }

        private static BufferedWriter gzipWriter(Path path) throws IOException {
            return new BufferedWriter(new OutputStreamWriter(
                    new GZIPOutputStream(Files.newOutputStream(path)), StandardCharsets.UTF_8));
        }

        public void onTick(Instrument instrument, ITick tick) throws JFException {
        }

        public void onBar(Instrument instrument, Period period, IBar askBar, IBar bidBar)
                throws JFException {
        }

        public void onMessage(IMessage message) throws JFException {
        }

        public void onAccount(IAccount account) throws JFException {
        }

        public void onStop() throws JFException {
        }
    }

    private static Instrument instrument(String pair) {
        if ("EUR/USD".equals(pair)) {
            return Instrument.EURUSD;
        }
        if ("GBP/USD".equals(pair)) {
            return Instrument.GBPUSD;
        }
        if ("USD/JPY".equals(pair)) {
            return Instrument.USDJPY;
        }
        throw new IllegalArgumentException("unsupported pair: " + pair);
    }

    private static final class Request {
        final String id;
        final String pair;
        final long fromMs;
        final long toMs;
        final String kind;

        Request(String id, String pair, long fromMs, long toMs, String kind) {
            this.id = id;
            this.pair = pair;
            this.fromMs = fromMs;
            this.toMs = toMs;
            this.kind = kind;
        }
    }

    private static final class TickRecord {
        final long time;
        final double bid;
        final double ask;
        final double bidVolume;
        final double askVolume;

        TickRecord(long time, double bid, double ask, double bidVolume, double askVolume) {
            this.time = time;
            this.bid = bid;
            this.ask = ask;
            this.bidVolume = bidVolume;
            this.askVolume = askVolume;
        }
    }
}
