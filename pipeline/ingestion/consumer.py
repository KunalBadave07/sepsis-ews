# pipeline/ingestion/consumer.py
import json
from datetime import datetime, timezone

from kafka import KafkaConsumer, KafkaProducer
from pydantic import ValidationError

from pipeline.validation.schema import VitalReading

BOOTSTRAP = "localhost:9092"
IN_TOPIC = "vitals.raw"
CLEAN_TOPIC = "vitals.clean"
DLQ_TOPIC = "vitals.dlq"


def run():
    consumer = KafkaConsumer(
        IN_TOPIC,
        bootstrap_servers=BOOTSTRAP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="validation-consumer",
    )
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print("Consumer running. Waiting for messages... (Ctrl+C to stop)")
    clean_count, dlq_count = 0, 0

    for msg in consumer:
        raw = msg.value
        try:
            reading = VitalReading(**raw)
            producer.send(CLEAN_TOPIC, value=reading.model_dump(mode="json"))
            clean_count += 1
            print(f"[CLEAN] {reading.patient_id} @ {reading.timestamp} "
                  f"HR={reading.heart_rate}")
        except ValidationError as e:
            dlq_count += 1
            producer.send(DLQ_TOPIC, value={
                "raw": raw,
                "error": str(e),
                "rejected_at": datetime.now(timezone.utc).isoformat(),
            })
            print(f"[DLQ] rejected: {e.errors()[0]['loc']} -> {e.errors()[0]['msg']}")

        if (clean_count + dlq_count) % 20 == 0:
            print(f"--- running totals: clean={clean_count} dlq={dlq_count} ---")


if __name__ == "__main__":
    run()