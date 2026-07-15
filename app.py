from flask import Flask, render_template, jsonify

app = Flask(__name__)


products = [
    {
        "id":1,
        "name":"Sports Shoes",
        "price":1500,
        "image":"https://images.unsplash.com/photo-1542291026-7eec264c27ff"
    },
    {
        "id":2,
        "name":"Smart Watch",
        "price":2500,
        "image":"https://images.unsplash.com/photo-1523275335684-37898b6baf30"
    },
    {
        "id":3,
        "name":"Headphones",
        "price":1200,
        "image":"https://images.unsplash.com/photo-1505740420928-5e560c06d30e"
    },
    {
        "id":4,
        "name":"T-Shirt",
        "price":800,
        "image":"https://images.unsplash.com/photo-1523381294911-8d3cead13475"
    },
    {
        "id":5,
        "name":"Hand Bag",
        "price":2200,
        "image":"https://images.unsplash.com/photo-1548036328-c9fa89d128fa"
    },
    {
        "id":6,
        "name":"Sunglasses",
        "price":900,
        "image":"https://images.unsplash.com/photo-1511499767150-a48a237f0083"
    }
]


@app.route("/")
def home():
    return render_template("index.html", products=products)



# REST API
@app.route("/api/products")
def api_products():
    return jsonify(products)



if __name__ == "__main__":
    app.run(debug=True)