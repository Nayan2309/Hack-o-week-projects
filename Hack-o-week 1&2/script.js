let cart = [];

function addProduct(){

let name=document.getElementById("name").value.trim();

let price=parseFloat(document.getElementById("price").value);

let qty=parseInt(document.getElementById("qty").value);

if(name==="" || isNaN(price) || isNaN(qty) || price<=0 || qty<=0){

alert("Please enter valid details.");

return;

}

let product={

name:name,

price:price,

quantity:qty

};

cart.push(product);

displayCart();

document.getElementById("name").value="";
document.getElementById("price").value="";
document.getElementById("qty").value="";

}

function displayCart(){

let body=document.getElementById("cartBody");

body.innerHTML="";

cart.forEach(function(item){

body.innerHTML+=`

<tr>

<td>${item.name}</td>

<td>₹${item.price}</td>

<td>${item.quantity}</td>

<td>₹${item.price*item.quantity}</td>

</tr>

`;

});

}

function calculateTotal(){

let subtotal=cart.reduce(function(total,item){

return total+(item.price*item.quantity);

},0);

let discount=0;

if(subtotal>=50000){

discount=subtotal*0.20;

}

else if(subtotal>=20000){

discount=subtotal*0.10;

}

else if(subtotal>=10000){

discount=subtotal*0.05;

}

let grand=subtotal-discount;

document.getElementById("subtotal").innerHTML="Subtotal : ₹"+subtotal.toFixed(2);

document.getElementById("discount").innerHTML="Discount : ₹"+discount.toFixed(2);

document.getElementById("grandTotal").innerHTML="Grand Total : ₹"+grand.toFixed(2);

}