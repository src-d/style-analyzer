/**
 * @preserve
 * jquery.layout 1.3.0 - Release Candidate 30.79
 * $Date: 2013-01-12 08:00:00 (Sat, 12 Jan 2013) $
 * $Rev: 303007 $
 *
 * Copyright (c) 2012 
 *   Fabrizio Balliano (http://www.fabrizioballiano.net)
 *   Kevin Dalman (http://allpro.net)
 *
 * Dual licensed under the GPL (http://www.gnu.org/licenses/gpl.html)
 * and MIT (http://www.opensource.org/licenses/mit-license.php) licenses.
 *
 * Changelog: http://layout.jquery-dev.net/changelog.cfm#1.3.0.rc30.79
 *
 * Docs: http://layout.jquery-dev.net/documentation.html
 * Tips: http://layout.jquery-dev.net/tips.html
 * Help: http://groups.google.com/group/jquery-ui-layout
 */

// NOTE: For best readability, view with a fixed-width font and tabs equal to 4-chars

;(function ($) {

// alias Math methods - used a lot!
var	min		= Math.min
,	max		= Math.max
,	round	= Math.floor

,	isStr	=  function (v) { return $.type(v) === "string"; }

	/**
	* @param {!Object}			Instance
	* @param {Array.<string>}	a_fn
	*/
,	runPluginCallbacks = function (Instance, a_fn) {
		if ($.isArray(a_fn))
			for (var i=0, c=a_fn.length; i<c; i++) {
				var fn = a_fn[i];
				try {
					if (isStr(fn)) // 'name' of a function
						fn = eval(fn);
					if ($.isFunction(fn))
						g(fn)( Instance );
				} catch (ex) {}
			}
		function g (f) { return f; }; // compiler hack
	}
;

/*
 *	$.layout.browser REPLACES removed $.browser, with extra data
 *	Parsing code here adapted from jQuery 1.8 $.browse
 */
var u = navigator.userAgent.toLowerCase()
,	m = /(chrome)[ \/]([\w.]+)/.exec( u )
	||	/(webkit)[ \/]([\w.]+)/.exec( u )
	||	/(opera)(?:.*version|)[ \/]([\w.]+)/.exec( u )
	||	/(msie) ([\w.]+)/.exec( u )
	||	u.indexOf("compatible") < 0 && /(mozilla)(?:.*? rv:([\w.]+)|)/.exec( u )
	||	[]
,	b = m[1] || ""
,	v = m[2] || 0
,	ie = b === "msie"
;
$.layout.browser = {
	version:	v
,	safari:		b === "webkit"	// webkit (NOT chrome) = safari
,	webkit:		b === "chrome"	// chrome = webkit
,	msie:		ie
,	isIE6:		ie && v == 6
	// ONLY IE reverts to old box-model - update for older jQ onReady
,	boxModel:	!ie || $.support.boxModel !== false
};
if (b) $.layout.browser[b] = true; // set CURRENT browser
/*	OLD versions of jQuery only set $.support.boxModel after page is loaded
 *	so if this is IE, use support.boxModel to test for quirks-mode (ONLY IE changes boxModel) */
if (ie) $(function(){ $.layout.browser.boxModel = $.support.boxModel; });


})( jQuery );
// END Layout - keep internal vars internal!

// START Plugins - shared wrapper, no global vars
(function ($) {


// add initialization method to Layout's onLoad array of functions
$.layout.onReady.push( $.layout.browserZoom._init );


})( jQuery );
