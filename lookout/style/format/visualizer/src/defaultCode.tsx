const defaultCode = String.raw`define( [
	"../core",
	"../core/stripAndCollapse",
	"./support",
	"../core/nodeName",
	"../var/isFunction",

	"../core/init"
], function( jQuery, stripAndCollapse, support, nodeName, isFunction ) {

"use strict";

var rreturn = /\r/g;

// Radios and checkboxes getter/setter
jQuery.each( [ "radio", "checkbox" ], function() {
	jQuery.valHooks[this] = {
		set: function( elem, value ) {
			if ( Array.isArray( value ) ) {
				return ( elem.checked = jQuery.inArray( jQuery( elem ).val(), value ) > -1 );
			}
		}
	};
	if ( !support.checkOn ) {
		jQuery.valHooks[ this ].get = function( elem ) {
			return elem.getAttribute( "value" ) === null ? "on" : elem.value;
		};
	}
} );

} );`;

export default defaultCode;
