from firebase_admin import initialize_app, firestore
from fastembed import TextEmbedding
from firebase_functions import https_fn, options
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector
from firebase_functions.firestore_fn import (
    on_document_written,
    Event,
    Change,
    DocumentSnapshot,
)

initialize_app()

embedding_model = TextEmbedding()

@on_document_written(document="products/{product_id}", region="europe-west3")
def on_company_written(event: Event[Change[DocumentSnapshot | None]]) -> None:
    try:
        previous_document = (event.data.before.to_dict() if event.data.before is not None else None)
        new_document = (event.data.after.to_dict() if event.data.after is not None else None)
        if new_document is None:
            return
        if previous_document is not None and previous_document['description'] == new_document['description']:
            return
        embeddings_list = list(embedding_model.embed([new_document['description']]))
        event.data.after.reference.set({'embedded_sentence': Vector(embeddings_list[0])}, merge=True)
    except Exception as e:
        print(e)


@https_fn.on_request(cors=options.CorsOptions(cors_origins="*", cors_methods=["post"]), region="europe-west3")
def knn_search(req: https_fn.Request) -> https_fn.Response:
    search_text = req.get_data(as_text=True)

    if search_text is None:
        return https_fn.Response(status=400)
    
    try:
        embeddings_list = list(embedding_model.embed([search_text]))
        company_vector = Vector(embeddings_list[0])
        result = firestore.client().collection('products').find_nearest(
            vector_field="embedded_sentence",
            query_vector=company_vector,
            distance_measure=DistanceMeasure.COSINE,
            limit=4
        ).get()

        return https_fn.Response(response=str([result[i].data() for i in range(len(result))]))
    except Exception as e:
        print(e)
        return https_fn.Response(status=500)
